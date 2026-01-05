#!/usr/bin/env python3
"""
Test script for the enhanced health check endpoint.
This script verifies that the health check properly validates database and AI service connectivity.
"""

import asyncio
import os
import sys
from typing import Dict

# Add the letta directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from letta.server.utils import (
    check_database_health,
    check_anthropic_health,
    check_all_ai_services,
)


async def test_database_health():
    """Test database health check with a real database connection."""
    print("=" * 70)
    print("Testing Database Health Check")
    print("=" * 70)
    
    # Use environment variable or default connection string
    pg_uri = os.getenv("LETTA_PG_URI", "postgresql+asyncpg://letta:letta@localhost:5432/letta")
    
    try:
        # Create async engine
        engine = create_async_engine(pg_uri)
        session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        async with session_factory() as session:
            result = await check_database_health(session)
            print(f"Service: {result.service}")
            print(f"Healthy: {result.healthy}")
            print(f"Error: {result.error}")
            print()
            
            if result.healthy:
                print("✅ Database health check PASSED")
            else:
                print("❌ Database health check FAILED")
            
        await engine.dispose()
        return result.healthy
        
    except Exception as e:
        print(f"❌ Database health check FAILED with exception: {e}")
        return False


async def test_ai_services():
    """Test AI service health checks."""
    print("\n" + "=" * 70)
    print("Testing AI Service Health Checks")
    print("=" * 70)
    
    # Test individual services
    services = {
        "Anthropic Direct": check_anthropic_health,
        "Vertex AI Anthropic": check_vertex_anthropic_health,
    }
    
    results = {}
    for service_name, check_func in services.items():
        print(f"\nTesting {service_name}...")
        result = await check_func()
        
        if result is None:
            print(f"  ⏭️  {service_name}: Not configured (skipped)")
            results[service_name] = "not_configured"
        elif result.healthy:
            print(f"  ✅ {service_name}: Healthy")
            results[service_name] = "healthy"
        else:
            print(f"  ❌ {service_name}: Unhealthy - {result.error}")
            results[service_name] = "unhealthy"
    
    return results


async def test_all_services_concurrent():
    """Test the check_all_ai_services function that runs checks concurrently."""
    print("\n" + "=" * 70)
    print("Testing Concurrent AI Service Health Checks")
    print("=" * 70)
    
    import time
    start_time = time.time()
    
    ai_services = await check_all_ai_services()
    
    elapsed_time = time.time() - start_time
    
    print(f"\nCompleted in {elapsed_time:.2f} seconds")
    print(f"Services checked: {len(ai_services)}")
    
    for service_name, service_health in ai_services.items():
        status = "✅ Healthy" if service_health.healthy else f"❌ Unhealthy: {service_health.error}"
        print(f"  - {service_name}: {status}")
    
    return ai_services


async def main():
    """Run all health check tests."""
    print("\n" + "=" * 70)
    print("LETTA HEALTH CHECK TEST SUITE")
    print("=" * 70)
    
    # Test database
    db_healthy = await test_database_health()
    
    # Test AI services
    ai_results = await test_ai_services()
    
    # Test concurrent execution
    ai_services = await test_all_services_concurrent()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 Database: {'✅ Healthy' if db_healthy else '❌ Unhealthy'}")
    
    configured_count = len([r for r in ai_results.values() if r != "not_configured"])
    healthy_count = len([r for r in ai_results.values() if r == "healthy"])
    
    print(f"📊 AI Services: {healthy_count}/{configured_count} healthy")
    
    if db_healthy and (configured_count == 0 or healthy_count == configured_count):
        print("\n🎉 All tests PASSED!")
        return 0
    else:
        print("\n⚠️  Some tests FAILED - check logs above")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
