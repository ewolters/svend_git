"""
Performance Benchmarking Management Command

Usage:
    python manage.py benchmark
    python manage.py benchmark --suite=database
    python manage.py benchmark --iterations=1000
    python manage.py benchmark --output=benchmark-results.json
"""

import time
from datetime import datetime
from typing import Any, Dict, List

from django.core.management.base import BaseCommand, CommandError
from django.db import connection
from django.utils import timezone


class BenchmarkResult:
    """Stores benchmark results."""

    def __init__(self, name: str):
        self.name = name
        self.iterations = 0
        self.total_time = 0.0
        self.min_time = float("inf")
        self.max_time = 0.0
        self.times: List[float] = []

    def add_result(self, duration: float):
        """Add a benchmark result."""
        self.iterations += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.times.append(duration)

    @property
    def avg_time(self) -> float:
        """Average execution time."""
        return self.total_time / self.iterations if self.iterations > 0 else 0

    @property
    def median_time(self) -> float:
        """Median execution time."""
        if not self.times:
            return 0
        sorted_times = sorted(self.times)
        n = len(sorted_times)
        if n % 2 == 0:
            return (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
        return sorted_times[n // 2]

    @property
    def ops_per_second(self) -> float:
        """Operations per second."""
        return self.iterations / self.total_time if self.total_time > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time": round(self.total_time, 6),
            "avg_time": round(self.avg_time, 6),
            "median_time": round(self.median_time, 6),
            "min_time": round(self.min_time, 6),
            "max_time": round(self.max_time, 6),
            "ops_per_second": round(self.ops_per_second, 2),
        }


class Command(BaseCommand):
    """Performance benchmarking command."""

    help = "Run performance benchmarks"

    def add_arguments(self, parser):
        parser.add_argument(
            "--suite",
            type=str,
            default="all",
            help="Benchmark suite to run: all, database, orm, cache, api (default: all)",
        )
        parser.add_argument(
            "--iterations",
            type=int,
            default=100,
            help="Number of iterations per benchmark (default: 100)",
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output file for results (JSON format)",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Show detailed output",
        )

    def handle(self, *args, **options):
        suite = options["suite"]
        iterations = options["iterations"]
        output_file = options["output"]
        verbose = options["verbose"]

        self.stdout.write(self.style.SUCCESS("\n" + "=" * 60))
        self.stdout.write(self.style.SUCCESS("Synara Core Performance Benchmarks"))
        self.stdout.write(self.style.SUCCESS("=" * 60))
        self.stdout.write(f"\nSuite: {suite}")
        self.stdout.write(f"Iterations: {iterations}")
        self.stdout.write(f"Timestamp: {datetime.now().isoformat()}\n")

        results = {}

        # Run benchmarks based on suite
        if suite in ["all", "database"]:
            results["database"] = self.benchmark_database(iterations, verbose)

        if suite in ["all", "orm"]:
            results["orm"] = self.benchmark_orm(iterations, verbose)

        if suite in ["all", "cache"]:
            results["cache"] = self.benchmark_cache(iterations, verbose)

        # Display results
        self.display_results(results)

        # Save to file if requested
        if output_file:
            self.save_results(results, output_file)

    def benchmark_database(self, iterations: int, verbose: bool) -> Dict[str, Any]:
        """Benchmark database operations."""
        self.stdout.write(self.style.HTTP_INFO("\n📊 Database Benchmarks"))
        self.stdout.write("-" * 60)

        results = {}

        # Simple query
        result = BenchmarkResult("Simple SELECT query")
        for _ in range(iterations):
            start = time.perf_counter()
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            duration = time.perf_counter() - start
            result.add_result(duration)
        results["simple_query"] = result.to_dict()
        if verbose:
            self.stdout.write(f"  Simple SELECT: {result.avg_time*1000:.2f}ms avg")

        # Multiple rows query
        result = BenchmarkResult("SELECT 100 rows")
        for _ in range(iterations):
            start = time.perf_counter()
            with connection.cursor() as cursor:
                cursor.execute("SELECT generate_series(1, 100)")
                cursor.fetchall()
            duration = time.perf_counter() - start
            result.add_result(duration)
        results["select_100_rows"] = result.to_dict()
        if verbose:
            self.stdout.write(f"  SELECT 100 rows: {result.avg_time*1000:.2f}ms avg")

        # Transaction
        result = BenchmarkResult("Transaction (BEGIN/COMMIT)")
        for _ in range(iterations):
            start = time.perf_counter()
            with connection.cursor() as cursor:
                cursor.execute("BEGIN")
                cursor.execute("SELECT 1")
                cursor.execute("COMMIT")
            duration = time.perf_counter() - start
            result.add_result(duration)
        results["transaction"] = result.to_dict()
        if verbose:
            self.stdout.write(f"  Transaction: {result.avg_time*1000:.2f}ms avg")

        self.stdout.write(self.style.SUCCESS("  ✓ Database benchmarks complete"))
        return results

    def benchmark_orm(self, iterations: int, verbose: bool) -> Dict[str, Any]:
        """Benchmark Django ORM operations."""
        self.stdout.write(self.style.HTTP_INFO("\n📦 ORM Benchmarks"))
        self.stdout.write("-" * 60)

        results = {}

        try:
            from django.contrib.auth import get_user_model

            User = get_user_model()

            # Count query
            result = BenchmarkResult("ORM count()")
            for _ in range(iterations):
                start = time.perf_counter()
                count = User.objects.count()
                duration = time.perf_counter() - start
                result.add_result(duration)
            results["orm_count"] = result.to_dict()
            if verbose:
                self.stdout.write(f"  ORM count(): {result.avg_time*1000:.2f}ms avg")

            # Exists query
            result = BenchmarkResult("ORM exists()")
            for _ in range(iterations):
                start = time.perf_counter()
                exists = User.objects.filter(is_active=True).exists()
                duration = time.perf_counter() - start
                result.add_result(duration)
            results["orm_exists"] = result.to_dict()
            if verbose:
                self.stdout.write(f"  ORM exists(): {result.avg_time*1000:.2f}ms avg")

            # Filter query
            result = BenchmarkResult("ORM filter()")
            for _ in range(iterations):
                start = time.perf_counter()
                users = list(User.objects.filter(is_active=True)[:10])
                duration = time.perf_counter() - start
                result.add_result(duration)
            results["orm_filter"] = result.to_dict()
            if verbose:
                self.stdout.write(f"  ORM filter(): {result.avg_time*1000:.2f}ms avg")

            self.stdout.write(self.style.SUCCESS("  ✓ ORM benchmarks complete"))
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"  ⚠ ORM benchmarks skipped: {e}"))

        return results

    def benchmark_cache(self, iterations: int, verbose: bool) -> Dict[str, Any]:
        """Benchmark cache operations."""
        self.stdout.write(self.style.HTTP_INFO("\n💾 Cache Benchmarks"))
        self.stdout.write("-" * 60)

        results = {}

        try:
            from django.core.cache import cache

            # Cache set
            result = BenchmarkResult("Cache set()")
            for i in range(iterations):
                start = time.perf_counter()
                cache.set(f"benchmark_key_{i}", "benchmark_value", timeout=60)
                duration = time.perf_counter() - start
                result.add_result(duration)
            results["cache_set"] = result.to_dict()
            if verbose:
                self.stdout.write(f"  Cache set(): {result.avg_time*1000:.2f}ms avg")

            # Cache get (hit)
            cache.set("benchmark_key", "benchmark_value", timeout=60)
            result = BenchmarkResult("Cache get() [hit]")
            for _ in range(iterations):
                start = time.perf_counter()
                value = cache.get("benchmark_key")
                duration = time.perf_counter() - start
                result.add_result(duration)
            results["cache_get_hit"] = result.to_dict()
            if verbose:
                self.stdout.write(f"  Cache get() [hit]: {result.avg_time*1000:.2f}ms avg")

            # Cache get (miss)
            result = BenchmarkResult("Cache get() [miss]")
            for _ in range(iterations):
                start = time.perf_counter()
                value = cache.get(f"nonexistent_key_{_}")
                duration = time.perf_counter() - start
                result.add_result(duration)
            results["cache_get_miss"] = result.to_dict()
            if verbose:
                self.stdout.write(f"  Cache get() [miss]: {result.avg_time*1000:.2f}ms avg")

            # Cache delete
            result = BenchmarkResult("Cache delete()")
            for i in range(iterations):
                cache.set(f"delete_key_{i}", "value", timeout=60)
                start = time.perf_counter()
                cache.delete(f"delete_key_{i}")
                duration = time.perf_counter() - start
                result.add_result(duration)
            results["cache_delete"] = result.to_dict()
            if verbose:
                self.stdout.write(f"  Cache delete(): {result.avg_time*1000:.2f}ms avg")

            # Cleanup
            for i in range(iterations):
                cache.delete(f"benchmark_key_{i}")

            self.stdout.write(self.style.SUCCESS("  ✓ Cache benchmarks complete"))
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"  ⚠ Cache benchmarks skipped: {e}"))

        return results

    def display_results(self, results: Dict[str, Dict[str, Any]]):
        """Display benchmark results."""
        self.stdout.write(self.style.SUCCESS("\n" + "=" * 60))
        self.stdout.write(self.style.SUCCESS("Benchmark Results Summary"))
        self.stdout.write(self.style.SUCCESS("=" * 60))

        for suite_name, suite_results in results.items():
            self.stdout.write(f"\n{suite_name.upper()}:")
            self.stdout.write("-" * 60)

            # Table header
            self.stdout.write(f"{'Operation':<30} {'Avg (ms)':<12} {'Median (ms)':<12} {'Ops/sec':<12}")
            self.stdout.write("-" * 60)

            # Table rows
            for benchmark_name, benchmark_data in suite_results.items():
                avg_ms = benchmark_data["avg_time"] * 1000
                median_ms = benchmark_data["median_time"] * 1000
                ops_per_sec = benchmark_data["ops_per_second"]

                self.stdout.write(
                    f"{benchmark_data['name']:<30} " f"{avg_ms:<12.3f} " f"{median_ms:<12.3f} " f"{ops_per_sec:<12.0f}"
                )

    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save results to JSON file."""
        import json

        output_data = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        self.stdout.write(self.style.SUCCESS(f"\n✓ Results saved to: {output_file}"))
