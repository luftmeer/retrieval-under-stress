from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/livez", include_in_schema=False)
def livez() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/readyz", include_in_schema=False)
def readyz(request: Request) -> dict[str, str]:
    resources = getattr(request.app.state, "resources", None)

    if resources is None:
        raise HTTPException(
            status_code=503,
            detail="Application resources are not initialized",
        )

    try:
        observed = resources.client.count(
            collection_name=resources.collection_name,
            exact=True,
            timeout=2,
        ).count
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Retrieval index is unavailable",
        ) from exc

    expected = len(resources.dataset.documents)

    if expected == 0 or observed != expected:
        raise HTTPException(
            status_code=503,
            detail="Retrieval index is incomplete",
        )

    return {"status": "ready"}