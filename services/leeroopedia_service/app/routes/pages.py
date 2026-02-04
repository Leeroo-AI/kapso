"""
Wiki pages endpoints.
"""

from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..auth.api_key import get_current_user, AuthenticatedUser
from ..services.wiki_reader import (
    get_wiki_reader,
    VALID_NAMESPACES,
    NAMESPACE_NAME_TO_DIR
)


router = APIRouter()


class PageListItem(BaseModel):
    """Page info in list response."""

    title: str
    namespace: str
    namespace_id: int


class PageListResponse(BaseModel):
    """Response for page listing."""

    pages: List[PageListItem]
    total: int
    namespace_filter: Optional[str]


class PageContentResponse(BaseModel):
    """Response for page content."""

    title: str
    namespace: str
    namespace_id: int
    content: str


class NamespaceInfo(BaseModel):
    """Namespace information."""

    id: int
    name: str


class NamespacesResponse(BaseModel):
    """Response for namespaces listing."""

    namespaces: List[NamespaceInfo]


@router.get("/v1/namespaces", response_model=NamespacesResponse)
async def list_namespaces(user: AuthenticatedUser = Depends(get_current_user)):
    """
    List all valid namespaces.

    Returns the available namespace names and IDs.
    """
    namespaces = [
        NamespaceInfo(id=ns_id, name=ns_name)
        for ns_id, ns_name in sorted(VALID_NAMESPACES.items())
    ]
    return NamespacesResponse(namespaces=namespaces)


@router.get("/v1/pages", response_model=PageListResponse)
async def list_pages(
    namespace: Optional[str] = Query(
        None,
        description="Filter by namespace (e.g., 'implementation', 'workflow')"
    ),
    user: AuthenticatedUser = Depends(get_current_user)
):
    """
    List all available wiki pages.

    Optionally filter by namespace.
    """
    # Validate namespace if provided
    if namespace:
        ns_lower = namespace.lower()
        if ns_lower not in NAMESPACE_NAME_TO_DIR:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_namespace",
                    "message": f"Invalid namespace: {namespace}",
                    "valid_namespaces": list(VALID_NAMESPACES.values())
                }
            )

    reader = get_wiki_reader()
    pages = reader.list_pages(namespace)

    return PageListResponse(
        pages=[
            PageListItem(
                title=p.title,
                namespace=p.namespace,
                namespace_id=p.namespace_id
            )
            for p in pages
        ],
        total=len(pages),
        namespace_filter=namespace
    )


@router.get("/v1/pages/{namespace}/{title:path}", response_model=PageContentResponse)
async def get_page(
    namespace: str,
    title: str,
    user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Get a specific wiki page by namespace and title.
    """
    # Validate namespace
    ns_lower = namespace.lower()
    if ns_lower not in NAMESPACE_NAME_TO_DIR:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_namespace",
                "message": f"Invalid namespace: {namespace}",
                "valid_namespaces": list(VALID_NAMESPACES.values())
            }
        )

    reader = get_wiki_reader()
    page = reader.get_page(namespace, title)

    if page is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "page_not_found",
                "message": f"Page not found: {namespace}:{title}"
            }
        )

    return PageContentResponse(
        title=page.title,
        namespace=page.namespace,
        namespace_id=page.namespace_id,
        content=page.content
    )
