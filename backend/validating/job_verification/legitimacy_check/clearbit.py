import httpx

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

CLEARBIT_AUTOCOMPLETE_URL = "https://autocomplete.clearbit.com/v1/companies/suggest"


async def clearbit_suggestions(company_name: str | None) -> list[dict]:
    """
    Return the full list of Clearbit autocomplete suggestions for a name,
    or [] on any failure / empty input. Used by verify_company.
    """
    if not company_name:
        return []
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=5) as client:
            r = await client.get(
                CLEARBIT_AUTOCOMPLETE_URL,
                params={"query": company_name},
            )
            results = r.json()
            return [s for s in results if isinstance(s, dict)] if isinstance(results, list) else []
    except Exception:
        return []


async def clearbit_lookup(company_name: str | None) -> dict | None:
    """Backward-compatible single-result helper."""
    suggestions = await clearbit_suggestions(company_name)
    return suggestions[0] if suggestions else None
