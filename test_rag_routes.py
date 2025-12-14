"""
Simple test to verify RAG routes are properly configured.
"""

from app.routes.rag import router

# Print all routes
print("RAG API Routes:")
print("=" * 80)

for route in router.routes:
    methods = ", ".join(route.methods) if hasattr(route, "methods") else "N/A"
    # Use path_format attribute for APIRoute objects
    path = route.path_format if hasattr(route, "path_format") else str(route)
    name = route.name if hasattr(route, "name") else "N/A"
    print(f"{methods:10} {path:50} ({name})")

print("=" * 80)
print(f"Total routes: {len(router.routes)}")
