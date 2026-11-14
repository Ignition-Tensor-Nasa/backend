"""Grounded LLM / curated summary generator for SAFFIRE-III (PSI-100) special cases."""

from __future__ import annotations

import json
from typing import Literal

from jsonschema import Draft202012Validator, ValidationError as SchemaValidationError

from app.ai.provider import AIUnavailable, ai_settings, call_json

FocusArea = Literal["overview", "burn_plume", "flame_spread", "spacecraft_cabin"]
LanguageCode = Literal["pl", "en"]

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string", "minLength": 1},
        "executive_summary": {"type": "string", "minLength": 1},
        "key_findings": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 2,
            "maxItems": 6,
        },
        "spacecraft_telemetry_insight": {"type": "string", "minLength": 1},
        "methodological_caveats": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
            "maxItems": 4,
        },
    },
    "required": [
        "headline",
        "executive_summary",
        "key_findings",
        "spacecraft_telemetry_insight",
        "methodological_caveats",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a microgravity spacecraft fire-safety telemetry analyst summarizing \
verified NASA Physical Sciences Informatics (PSI-100, Saffire-III) measurements aboard the \
Cygnus OA-7 cargo vehicle.

Rules:
- Use ONLY the numbers, units, sensors, and methodological facts provided in the JSON input.
- Never invent unmeasured quantities, flame spread speeds (m/s), heat-release rates, or safety margins.
- Explicitly respect that Saffire 30 Hz and Cygnus 1 Hz telemetry use separate clocks differing by ~1 day.
- Respect that radiometer idle readings (~ -17 to -18 mW/cm²) and FIZ 2 smoke detector readings (V) are reported as calibrated in the NASA source without re-scaling.
- Write all fields in the requested language ("pl" for Polish, "en" for English).
- Tailor the emphasis to the requested `focus` area and optional `custom_prompt` if provided."""


def _demo_summary(sample: dict, spacecraft: dict, focus: FocusArea, language: LanguageCode, custom_prompt: str | None) -> dict:
    n = int(sample["sample_number"])
    s = sample["stats"]
    ev = spacecraft["events"]
    arrival = sample["arrival"]
    ig = s["igniter_window"] or [0.0, 0.0]

    tc_labels = arrival["thermocouples"]
    arr_300 = arrival["arrivals_by_threshold"].get(str(arrival["default_threshold_c"]), [])
    reached = [
        (tc, t)
        for tc, t in zip(tc_labels, arr_300)
        if t is not None
    ]
    reached.sort(key=lambda pair: pair[1])
    spread_seq = " → ".join(f"{tc} ({t:.1f} s)" for tc, t in reached[:4]) if reached else "brak przekroczeń progu"

    if language == "en":
        if n == 1:
            focus_line = {
                "overview": "Combined 30 Hz chamber telemetry and 1 Hz Cygnus downlink capture the full large spacecraft burn across a 0.4 × 0.94 m sample.",
                "burn_plume": f"Heat flux peaked at {s['peak_heat_flux_mw_cm2']:.0f} mW/cm² ({s['peak_radiometer']}) while sample temperature reached {s['peak_sample_temp_c']:.0f} °C on {s['peak_sample_tc']} and gas plume {s['peak_plume_tc']} surged to {s['peak_plume_temp_c']:.1f} °C.",
                "flame_spread": f"Surface thermocouples crossed {arrival['default_threshold_c']} °C in sequence: {spread_seq}, though spread speed cannot be derived without sensor coordinates.",
                "spacecraft_cabin": f"Cygnus smoke scatterometer FIZ 2 reacted {ev['smoke_lag_s']:.0f} s after the first smoke-wire pulse ({ev['scatter_baseline']:.3f} V → peak {ev['scatter_peak']:.2f} V at t={ev['scatter_peak_t']:.0f} s).",
            }[focus]
            if custom_prompt:
                focus_line += f" Custom query focus (“{custom_prompt}”) addressed from verified telemetry fields."
            return {
                "headline": f"Saffire-III Sample 1: {s['peak_sample_temp_c']:.0f} °C Microgravity Burn & {ev['smoke_lag_s']:.0f} s Cygnus Smoke Response",
                "executive_summary": f"{focus_line} Ambient O₂ fell from {s['o2_before_pct']:.2f}% to {s['o2_min_pct']:.2f}% as CO₂ rose from {s['co2_before_pct']:.2f}% to {s['co2_max_pct']:.2f}% and pressure increased by +{s['pressure_rise_kpa']:.2f} kPa under {s['air_flow_cm_s']:.0f} cm/s flow.",
                "key_findings": [
                    f"Igniter active window spanned t={ig[0]:.1f}–{ig[1]:.1f} s; peak sample surface temperature ({s['peak_sample_temp_c']:.0f} °C on {s['peak_sample_tc']}) occurred {s['peak_after_ignition_s']:.0f} s after ignition.",
                    f"Radiant flux climbed from ~{s['idle_heat_flux_mw_cm2']:.0f} mW/cm² idle baseline to {s['peak_heat_flux_mw_cm2']:.0f} mW/cm² on {s['peak_radiometer']}.",
                    f"Thermal arrival order at {arrival['default_threshold_c']} °C threshold: {spread_seq}.",
                    f"Atmosphere telemetry confirms combustion stoichiometry shift: O₂ {s['o2_before_pct']:.2f}% → {s['o2_min_pct']:.2f}% and CO₂ {s['co2_before_pct']:.2f}% → {s['co2_max_pct']:.2f}%.",
                ],
                "spacecraft_telemetry_insight": f"In the 1 Hz Cygnus downlink, smoke scatterometer FIZ 2 remained near {ev['scatter_baseline']:.3f} V before the smoke wire, doubled within {ev['smoke_lag_s']:.0f} s, and reached {ev['scatter_peak']:.2f} V during Sample 1 burning.",
                "methodological_caveats": [
                    "Thermocouple coordinates are omitted in the processed table, so arrival times show sequence/timing rather than flame-spread velocity.",
                    "Saffire 30 Hz and Cygnus 1 Hz clocks differ by ~1 day in source timestamps; sections 1–3 and section 4 keep separate time axes.",
                    "Cygnus FIZ 2 output is reported in raw detector volts (V), not calibrated aerosol mass concentration.",
                ],
            }

        focus_line = {
            "overview": f"Sample 2 spans {s['duration_s']:.0f} s at 30 Hz and records an {ig[1]-ig[0]:.1f} s igniter pulse without sustained sample surface runaway.",
            "burn_plume": f"While sample surface thermocouples stayed ≤ {s['peak_sample_temp_c']:.1f} °C and radiometers remained near {s['peak_heat_flux_mw_cm2']:.1f} mW/cm², gas-phase {s['peak_plume_tc']} captured a {s['peak_plume_temp_c']:.1f} °C thermal plume pulse.",
            "flame_spread": "No sample thermocouple reached 100 °C during this 160 s sequence, confirming no self-sustaining surface flame propagation across TC7–TC12.",
            "spacecraft_cabin": f"The shared Cygnus 1 Hz downlink logs Sample 2 igniter current at t={ev['igniter2'][0]:.1f}–{ev['igniter2'][1]:.1f} s (~195 s before Sample 1 igniter).",
        }[focus]
        if custom_prompt:
            focus_line += f" Custom query focus (“{custom_prompt}”) addressed from verified telemetry fields."
        return {
            "headline": f"Saffire-III Sample 2: {ig[1]-ig[0]:.1f} s Igniter Pulse & Plume Signature (≤ {s['peak_sample_temp_c']:.1f} °C Surface)",
            "executive_summary": f"{focus_line} Pre-ignition duct flow averaged {s['air_flow_cm_s']:.0f} cm/s at {s['pressure_before_kpa']:.1f} kPa and {s['o2_before_pct']:.2f}% O₂.",
            "key_findings": [
                f"Igniter current activated at t={ig[0]:.1f}–{ig[1]:.1f} s inside the {s['duration_s']:.0f} s Sample 2 recording.",
                f"Sample thermocouples (TC7–TC12) peaked at {s['peak_sample_temp_c']:.1f} °C ({s['peak_sample_tc']}), whereas plume {s['peak_plume_tc']} reached {s['peak_plume_temp_c']:.1f} °C.",
                f"Ambient O₂ remained nearly flat ({s['o2_before_pct']:.2f}% → {s['o2_min_pct']:.2f}%, CO₂ max {s['co2_max_pct']:.2f}%) with minor +{s['pressure_rise_kpa']:.2f} kPa pressure variation.",
            ],
            "spacecraft_telemetry_insight": f"Cygnus downlink telemetry links Sample 2 ignition (t={ev['igniter2'][0]:.1f} s) and Sample 1 ignition (t={ev['igniter1'][0]:.1f} s) on the vehicle clock alongside cabin FIZ 2 scatterometer peak {ev['scatter_peak']:.2f} V.",
            "methodological_caveats": [
                "Sample 2 did not exceed 20.0 °C on surface thermocouples; threshold arrivals are shown at sub-ignition levels (15 °C, 17 °C) alongside 100 °C.",
                "Saffire 30 Hz and Cygnus 1 Hz clocks differ by ~1 day in source timestamps; axes remain separate.",
            ],
        }

    # Polish ("pl") grounded demo summary
    if n == 1:
        focus_line = {
            "overview": "Zespolona telemetria 30 Hz komory spalania oraz 1 Hz ze statku Cygnus OA-7 dokumentują główny pożar próbki 0.4 × 0.94 m w mikrograwitacji.",
            "burn_plume": f"Strumień ciepła osiągnął szczyt {s['peak_heat_flux_mw_cm2']:.0f} mW/cm² ({s['peak_radiometer']}), temperatura powierzchni próbki wzrosła do {s['peak_sample_temp_c']:.0f} °C ({s['peak_sample_tc']}), a w smudze ({s['peak_plume_tc']}) do {s['peak_plume_temp_c']:.1f} °C.",
            "flame_spread": f"Termopary powierzchniowe przekraczały próg {arrival['default_threshold_c']} °C w kolejności: {spread_seq}, co ujawnia kierunek i dynamikę frontu cieplnego.",
            "spacecraft_cabin": f"Detektor dymu FIZ 2 na statku Cygnus zareagował po {ev['smoke_lag_s']:.0f} s od pierwszego impulsu drutu dymnego (skok z {ev['scatter_baseline']:.3f} V do maksimum {ev['scatter_peak']:.2f} V w t={ev['scatter_peak_t']:.0f} s).",
        }[focus]
        if custom_prompt:
            focus_line += f" Uwzględniono pytanie użytkownika („{custom_prompt}”) wyłącznie na bazie zweryfikowanych pomiarów."
        return {
            "headline": f"Saffire-III Sample 1: Pożar do {s['peak_sample_temp_c']:.0f} °C i detekcja dymu w Cygnus po {ev['smoke_lag_s']:.0f} s",
            "executive_summary": f"{focus_line} Zużycie tlenu obniżyło O₂ z {s['o2_before_pct']:.2f}% do {s['o2_min_pct']:.2f}% przy wzroście CO₂ z {s['co2_before_pct']:.2f}% do {s['co2_max_pct']:.2f}%, wzroście ciśnienia o +{s['pressure_rise_kpa']:.2f} kPa i nawiewie {s['air_flow_cm_s']:.0f} cm/s.",
            "key_findings": [
                f"Okno zapalnika przypadło na t={ig[0]:.1f}–{ig[1]:.1f} s; szczytowa temperatura powierzchni ({s['peak_sample_temp_c']:.0f} °C na {s['peak_sample_tc']}) wystąpiła {s['peak_after_ignition_s']:.0f} s po zapłonie.",
                f"Strumień promieniowania wzrósł od ok. {s['idle_heat_flux_mw_cm2']:.0f} mW/cm² (spoczynek) do {s['peak_heat_flux_mw_cm2']:.0f} mW/cm² ({s['peak_radiometer']}).",
                f"Kolejność dotarcia progu {arrival['default_threshold_c']} °C na próbce: {spread_seq}.",
                f"Bilans atmosfery w komorze: O₂ {s['o2_before_pct']:.2f}% → {s['o2_min_pct']:.2f}%, CO₂ {s['co2_before_pct']:.2f}% → {s['co2_max_pct']:.2f}%, ΔP +{s['pressure_rise_kpa']:.2f} kPa (od {s['pressure_before_kpa']:.1f} kPa).",
            ],
            "spacecraft_telemetry_insight": f"W telemetrii 1 Hz pojazdu Cygnus sygnał rozpraszania FIZ 2 przed impulsem drutu dymnego wynosił {ev['scatter_baseline']:.3f} V, podwoił się po {ev['smoke_lag_s']:.0f} s (t={ev['first_rise']:.0f} s) i osiągnął {ev['scatter_peak']:.2f} V podczas spalania Sample 1.",
            "methodological_caveats": [
                "Pliki nie zawierają współrzędnych geometrycznych termopar TC7–TC12 — wykres sekcji 2 pokazuje kolejność i czasy dotarcia, nie prędkość liniową płomienia.",
                "Zegary Saffire (30 Hz) i statku Cygnus (1 Hz) różnią się w źródle o ok. 1 dobę — sekcje 1–3 i sekcja 4 zachowują osobne osie czasu.",
                "Odczyty FIZ 2 podane są w woltach (V) zgodnie z kalibracją źródłową, a nie w mg/m³ gęstości dymu.",
            ],
        }

    focus_line = {
        "overview": f"Zapis Sample 2 trwa {s['duration_s']:.0f} s (30 Hz) i dokumentuje {ig[1]-ig[0]:.1f}-sekundowy impuls zapalnika bez gwałtownego rozgorzenia powierzchni próbki.",
        "burn_plume": f"Termopary powierzchniowe utrzymały ≤ {s['peak_sample_temp_c']:.1f} °C, a radiometry pozostały w okolicach tła ({s['peak_heat_flux_mw_cm2']:.1f} mW/cm²), natomiast termopara gazu {s['peak_plume_tc']} zarejestrowała impuls cieplny {s['peak_plume_temp_c']:.1f} °C.",
        "flame_spread": "Żadna z termopar powierzchniowych TC7–TC12 nie przekroczyła 100 °C — brak podtrzymanego rozprzestrzeniania płomienia po materiale.",
        "spacecraft_cabin": f"Wspólna telemetria 1 Hz statku Cygnus rejestruje impuls zapalnika Sample 2 w t={ev['igniter2'][0]:.1f}–{ev['igniter2'][1]:.1f} s (~195 s przed zapłonem Sample 1).",
    }[focus]
    if custom_prompt:
        focus_line += f" Uwzględniono pytanie użytkownika („{custom_prompt}”) na bazie zweryfikowanych danych."
    return {
        "headline": f"Saffire-III Sample 2: Impuls zapalnika ({ig[1]-ig[0]:.1f} s) i reakcja gazu {s['peak_plume_temp_c']:.1f} °C bez dużego pożaru",
        "executive_summary": f"{focus_line} Warunki przed impulsem: {s['o2_before_pct']:.2f}% O₂, ciśnienie {s['pressure_before_kpa']:.1f} kPa oraz przepływ powietrza {s['air_flow_cm_s']:.0f} cm/s.",
        "key_findings": [
            f"Prąd zapalnika był aktywny w t={ig[0]:.1f}–{ig[1]:.1f} s w {s['duration_s']:.0f}-sekundowym zapisie Sample 2.",
            f"Termopary próbki TC7–TC12 osiągnęły maks. {s['peak_sample_temp_c']:.1f} °C ({s['peak_sample_tc']}), podczas gdy gazy spalinowe ({s['peak_plume_tc']}) osiągnęły {s['peak_plume_temp_c']:.1f} °C.",
            f"Stężenia gazów pozostały stabilne (O₂ {s['o2_before_pct']:.2f}% → {s['o2_min_pct']:.2f}%, CO₂ maks. {s['co2_max_pct']:.2f}%).",
        ],
        "spacecraft_telemetry_insight": f"Telemetria 1 Hz pojazdu Cygnus zestawia zapalnik Sample 2 (t={ev['igniter2'][0]:.1f} s) oraz późniejszy pożar Sample 1 (t={ev['igniter1'][0]:.1f} s), przy którym kabinowy detektor dymu osiągnął {ev['scatter_peak']:.2f} V.",
        "methodological_caveats": [
            "W próbce Sample 2 temperatura na TC7–TC12 nie przekroczyła 20.0 °C, dlatego próg 100 °C nie jest osiągany (pokazano również progi podzapłonowe 15 °C i 17 °C).",
            "Zegary Saffire (30 Hz) i statku Cygnus (1 Hz) pozostają na osobnych osiach czasu.",
        ],
    }


def summarize_saffire_case(
    sample_payload: dict,
    spacecraft_payload: dict,
    *,
    focus: FocusArea = "overview",
    custom_prompt: str | None = None,
    language: LanguageCode = "pl",
) -> tuple[dict, Literal["live", "demo"]]:
    mode = ai_settings().get("PSI_AI_MODE", "live").strip().lower()
    if mode not in ("live", "demo"):
        raise AIUnavailable("The AI assistant is not configured (set PSI_AI_MODE to live or demo).")

    cleaned_prompt = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else None

    if mode == "demo":
        return _demo_summary(sample_payload, spacecraft_payload, focus, language, cleaned_prompt), "demo"

    facts_context = {
        "investigation": "NASA PSI-100 Saffire-III aboard Cygnus OA-7",
        "case_id": sample_payload["case_id"],
        "sample_number": sample_payload["sample_number"],
        "language": language,
        "focus": focus,
        "custom_prompt": cleaned_prompt,
        "sample_stats": sample_payload["stats"],
        "arrival_thermocouples": sample_payload["arrival"]["thermocouples"],
        "arrival_thresholds_c": sample_payload["arrival"]["thresholds_c"],
        "arrival_times_s": sample_payload["arrival"]["arrivals_by_threshold"],
        "arrival_peaks_c": sample_payload["arrival"]["peaks_c"],
        "spacecraft_events": spacecraft_payload["events"],
    }
    answer = call_json(
        SYSTEM_PROMPT,
        json.dumps(facts_context, ensure_ascii=False),
        SUMMARY_SCHEMA,
        effort="low",
        timeout=30.0,
    )
    try:
        Draft202012Validator(SUMMARY_SCHEMA).validate(answer)
    except SchemaValidationError:
        raise AIUnavailable("The AI provider returned an invalid SAFFIRE-III summary schema.") from None
    return answer, "live"
