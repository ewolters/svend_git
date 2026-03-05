"""
Synara SIEM Export Module (AUD-001 §14, SEC-002 §8)
===================================================

SIEM (Security Information and Event Management) integration for
centralized log aggregation, correlation, and security monitoring.

Standard:     AUD-001 §14 (SIEM Integration), SEC-002 §8 (Security Operations)
Compliance:   SOC 2 CC7.2, ISO 27001 A.12.4, NIST SP 800-53 AU-6
Version:      1.0.0
Last Updated: 2025-12-03

Provides:
- Multi-format export (CEF, LEEF, JSON, Syslog)
- Batch and streaming export modes
- Event filtering based on siem_forward flag
- Severity mapping to SIEM standards
- Correlation ID propagation
- TLS transport security (ENC-001 §6)

Supported SIEM Platforms:
- Splunk (HEC endpoint)
- Elastic/ELK Stack
- AWS Security Hub
- Generic Syslog (RFC 5424)
- Generic HTTP/JSON

References:
- AUD-001: Audit Logging Standard
- SEC-002: Security Operations Standard
- ENC-001: Encryption Controls
"""

from __future__ import annotations

import json
import logging
import socket
import ssl
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import requests
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Configuration (SEC-002 §8)
# =============================================================================


class SIEMFormat(Enum):
    """Supported SIEM export formats."""

    CEF = "cef"  # Common Event Format (ArcSight)
    LEEF = "leef"  # Log Event Extended Format (IBM QRadar)
    JSON = "json"  # Generic JSON
    SYSLOG = "syslog"  # RFC 5424 Syslog


class SIEMSeverity(Enum):
    """SIEM severity levels mapped to standards."""

    UNKNOWN = 0
    LOW = 1
    MEDIUM = 4
    HIGH = 7
    CRITICAL = 10


# Mapping from Synara severity to SIEM severity
SEVERITY_MAP = {
    "info": SIEMSeverity.LOW,
    "low": SIEMSeverity.LOW,
    "medium": SIEMSeverity.MEDIUM,
    "high": SIEMSeverity.HIGH,
    "critical": SIEMSeverity.CRITICAL,
}


# =============================================================================
# Data Classes (AUD-001 §4)
# =============================================================================


@dataclass
class SIEMEvent:
    """
    Normalized SIEM event for export.

    Captures the essential fields for SIEM integration per AUD-001 §4.

    Attributes:
        event_id: Unique event identifier
        correlation_id: CTG correlation ID
        timestamp: Event timestamp (UTC)
        event_name: Synara event name (domain.entity.action)
        severity: Event severity
        tenant_id: Tenant identifier
        actor_id: Actor who triggered the event
        actor_ip: Actor's IP address
        resource_type: Type of resource affected
        resource_id: ID of resource affected
        action: Human-readable action description
        status: Event status (success/failure/blocked/error)
        payload: Full event payload
    """

    event_id: str
    correlation_id: str
    timestamp: datetime
    event_name: str
    severity: SIEMSeverity
    tenant_id: str
    actor_id: str | None = None
    actor_ip: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    action: str | None = None
    status: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_cef(self, vendor: str = "Synara", product: str = "QMS", version: str = "1.0") -> str:
        """
        Convert to Common Event Format (CEF).

        CEF format: CEF:0|Vendor|Product|Version|EventID|Name|Severity|Extension

        Returns:
            CEF formatted string
        """

        # Sanitize fields for CEF (escape pipes and backslashes)
        def sanitize(value: str) -> str:
            if value is None:
                return ""
            return str(value).replace("\\", "\\\\").replace("|", "\\|")

        # Build extension fields
        extensions = []
        extensions.append(f"rt={int(self.timestamp.timestamp() * 1000)}")
        extensions.append("dvchost=synara")
        extensions.append(f"cs1={sanitize(self.correlation_id)}")
        extensions.append("cs1Label=CorrelationID")
        extensions.append(f"cs2={sanitize(self.tenant_id)}")
        extensions.append("cs2Label=TenantID")

        if self.actor_id:
            extensions.append(f"suser={sanitize(self.actor_id)}")
        if self.actor_ip:
            extensions.append(f"src={sanitize(self.actor_ip)}")
        if self.resource_type:
            extensions.append(f"cs3={sanitize(self.resource_type)}")
            extensions.append("cs3Label=ResourceType")
        if self.resource_id:
            extensions.append(f"cs4={sanitize(self.resource_id)}")
            extensions.append("cs4Label=ResourceID")
        if self.status:
            extensions.append(f"outcome={sanitize(self.status)}")
        if self.action:
            extensions.append(f"act={sanitize(self.action)}")

        ext_str = " ".join(extensions)

        return (
            f"CEF:0|{sanitize(vendor)}|{sanitize(product)}|{sanitize(version)}|"
            f"{sanitize(self.event_id)}|{sanitize(self.event_name)}|"
            f"{self.severity.value}|{ext_str}"
        )

    def to_leef(self, vendor: str = "Synara", product: str = "QMS", version: str = "1.0") -> str:
        """
        Convert to Log Event Extended Format (LEEF).

        LEEF format: LEEF:Version|Vendor|Product|Version|EventID|Extension

        Returns:
            LEEF formatted string
        """

        def sanitize(value: str) -> str:
            if value is None:
                return ""
            return str(value).replace("\t", " ").replace("\n", " ")

        # Build key-value pairs
        fields = []
        fields.append(f"devTime={self.timestamp.strftime('%b %d %Y %H:%M:%S')}")
        fields.append(f"sev={self.severity.value}")
        fields.append(f"correlationId={sanitize(self.correlation_id)}")
        fields.append(f"tenantId={sanitize(self.tenant_id)}")
        fields.append(f"eventName={sanitize(self.event_name)}")

        if self.actor_id:
            fields.append(f"usrName={sanitize(self.actor_id)}")
        if self.actor_ip:
            fields.append(f"src={sanitize(self.actor_ip)}")
        if self.resource_type:
            fields.append(f"resourceType={sanitize(self.resource_type)}")
        if self.resource_id:
            fields.append(f"resourceId={sanitize(self.resource_id)}")
        if self.status:
            fields.append(f"outcome={sanitize(self.status)}")
        if self.action:
            fields.append(f"action={sanitize(self.action)}")

        ext_str = "\t".join(fields)

        return (
            f"LEEF:1.0|{sanitize(vendor)}|{sanitize(product)}|{sanitize(version)}|{sanitize(self.event_id)}|{ext_str}"
        )

    def to_json(self) -> str:
        """
        Convert to JSON format.

        Returns:
            JSON formatted string
        """
        data = {
            "event_id": self.event_id,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "event_name": self.event_name,
            "severity": self.severity.name.lower(),
            "severity_level": self.severity.value,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "actor_ip": self.actor_ip,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "status": self.status,
            "payload": self.payload,
            "_source": "synara",
            "_version": "1.0",
        }
        return json.dumps(data, default=str)

    def to_syslog(self, facility: int = 16, hostname: str = "synara") -> str:
        """
        Convert to RFC 5424 Syslog format.

        Args:
            facility: Syslog facility (default: 16 = local0)
            hostname: Hostname to include in message

        Returns:
            Syslog formatted string
        """
        # Map severity to syslog severity (0-7)
        syslog_severity_map = {
            SIEMSeverity.UNKNOWN: 6,  # Informational
            SIEMSeverity.LOW: 6,  # Informational
            SIEMSeverity.MEDIUM: 4,  # Warning
            SIEMSeverity.HIGH: 3,  # Error
            SIEMSeverity.CRITICAL: 2,  # Critical
        }
        syslog_severity = syslog_severity_map.get(self.severity, 6)

        # Calculate PRI
        pri = (facility * 8) + syslog_severity

        # Format timestamp (RFC 5424)
        timestamp = self.timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        # Build structured data
        sd = (
            f'[synara@49152 correlationId="{self.correlation_id}" '
            f'tenantId="{self.tenant_id}" eventName="{self.event_name}"'
        )
        if self.actor_id:
            sd += f' actorId="{self.actor_id}"'
        if self.resource_id:
            sd += f' resourceId="{self.resource_id}"'
        sd += "]"

        # Build message
        msg = self.action or self.event_name

        return f"<{pri}>1 {timestamp} {hostname} synara - {self.event_id} {sd} {msg}"


# =============================================================================
# SIEM Exporters (SEC-002 §8.1)
# =============================================================================


class SIEMExporter(ABC):
    """
    Abstract base class for SIEM exporters.

    Provides common interface for exporting events to different SIEM platforms.
    """

    @abstractmethod
    def export(self, events: list[SIEMEvent]) -> bool:
        """Export events to SIEM platform."""
        pass

    @abstractmethod
    def export_single(self, event: SIEMEvent) -> bool:
        """Export single event to SIEM platform."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test connection to SIEM platform."""
        pass


class HTTPExporter(SIEMExporter):
    """
    Generic HTTP/HTTPS SIEM exporter.

    Supports JSON payload to any HTTP endpoint.
    """

    def __init__(
        self,
        endpoint_url: str,
        auth_token: str | None = None,
        auth_header: str = "Authorization",
        verify_ssl: bool = True,
        timeout: int = 30,
        format: SIEMFormat = SIEMFormat.JSON,
    ):
        self.endpoint_url = endpoint_url
        self.auth_token = auth_token
        self.auth_header = auth_header
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.format = format

    def _get_headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Synara-SIEM-Exporter/1.0",
        }
        if self.auth_token:
            headers[self.auth_header] = f"Bearer {self.auth_token}"
        return headers

    def _format_event(self, event: SIEMEvent) -> str:
        """Format event according to configured format."""
        if self.format == SIEMFormat.CEF:
            return event.to_cef()
        elif self.format == SIEMFormat.LEEF:
            return event.to_leef()
        elif self.format == SIEMFormat.SYSLOG:
            return event.to_syslog()
        else:
            return event.to_json()

    def export(self, events: list[SIEMEvent]) -> bool:
        """Export batch of events."""
        try:
            if self.format == SIEMFormat.JSON:
                payload = [json.loads(e.to_json()) for e in events]
            else:
                payload = [self._format_event(e) for e in events]

            response = requests.post(
                self.endpoint_url,
                json=payload,
                headers=self._get_headers(),
                verify=self.verify_ssl,
                timeout=self.timeout,
            )
            response.raise_for_status()

            logger.info(f"[SIEM] Exported {len(events)} events to {self.endpoint_url}")
            return True

        except requests.RequestException as e:
            logger.error(f"[SIEM] Failed to export events: {e}")
            return False

    def export_single(self, event: SIEMEvent) -> bool:
        """Export single event."""
        return self.export([event])

    def test_connection(self) -> bool:
        """Test connection to HTTP endpoint."""
        try:
            response = requests.get(
                self.endpoint_url,
                headers=self._get_headers(),
                verify=self.verify_ssl,
                timeout=self.timeout,
            )
            return response.status_code < 500
        except requests.RequestException:
            return False


class SplunkHECExporter(HTTPExporter):
    """
    Splunk HTTP Event Collector (HEC) exporter.

    Specialized for Splunk's HEC endpoint format.
    """

    def __init__(
        self,
        hec_url: str,
        hec_token: str,
        index: str = "main",
        source: str = "synara",
        sourcetype: str = "synara:audit",
        verify_ssl: bool = True,
        timeout: int = 30,
    ):
        super().__init__(
            endpoint_url=hec_url,
            auth_token=hec_token,
            auth_header="Authorization",
            verify_ssl=verify_ssl,
            timeout=timeout,
            format=SIEMFormat.JSON,
        )
        self.index = index
        self.source = source
        self.sourcetype = sourcetype

    def _get_headers(self) -> dict[str, str]:
        """Build Splunk HEC headers."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Splunk {self.auth_token}",
        }

    def export(self, events: list[SIEMEvent]) -> bool:
        """Export events to Splunk HEC."""
        try:
            # Splunk HEC expects newline-delimited JSON
            hec_events = []
            for event in events:
                hec_event = {
                    "time": event.timestamp.timestamp(),
                    "source": self.source,
                    "sourcetype": self.sourcetype,
                    "index": self.index,
                    "event": json.loads(event.to_json()),
                }
                hec_events.append(json.dumps(hec_event))

            payload = "\n".join(hec_events)

            response = requests.post(
                self.endpoint_url,
                data=payload,
                headers=self._get_headers(),
                verify=self.verify_ssl,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            if result.get("text") == "Success":
                logger.info(f"[SIEM/Splunk] Exported {len(events)} events")
                return True
            else:
                logger.error(f"[SIEM/Splunk] Export failed: {result}")
                return False

        except requests.RequestException as e:
            logger.error(f"[SIEM/Splunk] Failed to export events: {e}")
            return False


class SyslogExporter(SIEMExporter):
    """
    RFC 5424 Syslog exporter.

    Supports UDP and TCP (with optional TLS) transport.
    """

    def __init__(
        self,
        host: str,
        port: int = 514,
        protocol: str = "udp",
        use_tls: bool = False,
        timeout: int = 30,
        facility: int = 16,
    ):
        self.host = host
        self.port = port
        self.protocol = protocol.lower()
        self.use_tls = use_tls
        self.timeout = timeout
        self.facility = facility
        self._socket = None

    def _get_socket(self) -> socket.socket:
        """Get or create socket connection."""
        if self._socket is not None:
            return self._socket

        if self.protocol == "udp":
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)

            if self.use_tls:
                context = ssl.create_default_context()
                self._socket = context.wrap_socket(
                    self._socket,
                    server_hostname=self.host,
                )

            self._socket.connect((self.host, self.port))

        return self._socket

    def export(self, events: list[SIEMEvent]) -> bool:
        """Export events via Syslog."""
        try:
            sock = self._get_socket()
            success_count = 0

            for event in events:
                message = event.to_syslog(facility=self.facility)
                data = (message + "\n").encode("utf-8")

                if self.protocol == "udp":
                    sock.sendto(data, (self.host, self.port))
                else:
                    sock.send(data)

                success_count += 1

            logger.info(f"[SIEM/Syslog] Exported {success_count} events to {self.host}:{self.port}")
            return True

        except OSError as e:
            logger.error(f"[SIEM/Syslog] Failed to export events: {e}")
            self._socket = None  # Reset socket on error
            return False

    def export_single(self, event: SIEMEvent) -> bool:
        """Export single event via Syslog."""
        return self.export([event])

    def test_connection(self) -> bool:
        """Test Syslog connection."""
        try:
            sock = self._get_socket()
            return sock is not None
        except OSError:
            return False

    def close(self):
        """Close socket connection."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None


# =============================================================================
# SIEM Manager (SEC-002 §8.2)
# =============================================================================


class SIEMManager:
    """
    Central manager for SIEM export operations.

    Handles event filtering, format conversion, and export to configured
    SIEM platforms.
    """

    def __init__(self):
        self.exporters: list[SIEMExporter] = []
        self._initialized = False

    def initialize(self):
        """Initialize SIEM exporters from settings."""
        if self._initialized:
            return

        siem_config = getattr(settings, "SIEM_CONFIG", {})

        for name, config in siem_config.items():
            try:
                exporter = self._create_exporter(name, config)
                if exporter:
                    self.exporters.append(exporter)
                    logger.info(f"[SIEM] Initialized exporter: {name}")
            except Exception as e:
                logger.error(f"[SIEM] Failed to initialize exporter {name}: {e}")

        self._initialized = True

    def _create_exporter(self, name: str, config: dict[str, Any]) -> SIEMExporter | None:
        """Create exporter from configuration."""
        exporter_type = config.get("type", "http").lower()

        if exporter_type == "splunk":
            return SplunkHECExporter(
                hec_url=config["url"],
                hec_token=config["token"],
                index=config.get("index", "main"),
                source=config.get("source", "synara"),
                sourcetype=config.get("sourcetype", "synara:audit"),
                verify_ssl=config.get("verify_ssl", True),
                timeout=config.get("timeout", 30),
            )
        elif exporter_type == "syslog":
            return SyslogExporter(
                host=config["host"],
                port=config.get("port", 514),
                protocol=config.get("protocol", "udp"),
                use_tls=config.get("use_tls", False),
                timeout=config.get("timeout", 30),
                facility=config.get("facility", 16),
            )
        elif exporter_type == "http":
            return HTTPExporter(
                endpoint_url=config["url"],
                auth_token=config.get("token"),
                auth_header=config.get("auth_header", "Authorization"),
                verify_ssl=config.get("verify_ssl", True),
                timeout=config.get("timeout", 30),
                format=SIEMFormat(config.get("format", "json")),
            )
        else:
            logger.warning(f"[SIEM] Unknown exporter type: {exporter_type}")
            return None

    def export_audit_entry(self, entry) -> bool:
        """
        Export a SysLogEntry to SIEM.

        Args:
            entry: SysLogEntry instance

        Returns:
            bool: True if exported successfully to at least one exporter
        """
        from syn.audit.events import AUDIT_EVENTS

        # Check if event should be forwarded to SIEM
        event_config = AUDIT_EVENTS.get(entry.event_name, {})
        if not event_config.get("siem_forward", False):
            return True  # Not an error, just not configured for SIEM

        # Convert to SIEMEvent
        severity_str = event_config.get("siem_severity", "medium")
        siem_event = SIEMEvent(
            event_id=str(entry.id),
            correlation_id=str(entry.correlation_id) if entry.correlation_id else str(uuid4()),
            timestamp=entry.timestamp,
            event_name=entry.event_name,
            severity=SEVERITY_MAP.get(severity_str, SIEMSeverity.MEDIUM),
            tenant_id=entry.tenant_id,
            actor_id=entry.actor,
            action=entry.payload.get("action"),
            status=entry.payload.get("status"),
            resource_type=entry.payload.get("resource_type"),
            resource_id=entry.payload.get("resource_id"),
            payload=entry.payload,
        )

        return self.export_event(siem_event)

    def export_event(self, event: SIEMEvent) -> bool:
        """
        Export a SIEMEvent to all configured exporters.

        Args:
            event: SIEMEvent instance

        Returns:
            bool: True if exported successfully to at least one exporter
        """
        self.initialize()

        if not self.exporters:
            logger.debug("[SIEM] No exporters configured, skipping export")
            return True

        success = False
        for exporter in self.exporters:
            try:
                if exporter.export_single(event):
                    success = True
            except Exception as e:
                logger.error(f"[SIEM] Export failed for {type(exporter).__name__}: {e}")

        return success

    def export_batch(self, events: list[SIEMEvent]) -> bool:
        """
        Export batch of events to all configured exporters.

        Args:
            events: List of SIEMEvent instances

        Returns:
            bool: True if exported successfully to at least one exporter
        """
        self.initialize()

        if not self.exporters:
            logger.debug("[SIEM] No exporters configured, skipping export")
            return True

        if not events:
            return True

        success = False
        for exporter in self.exporters:
            try:
                if exporter.export(events):
                    success = True
            except Exception as e:
                logger.error(f"[SIEM] Batch export failed for {type(exporter).__name__}: {e}")

        return success

    def test_connections(self) -> dict[str, bool]:
        """
        Test connections to all configured SIEM platforms.

        Returns:
            Dict mapping exporter names to connection status
        """
        self.initialize()

        results = {}
        for i, exporter in enumerate(self.exporters):
            name = f"{type(exporter).__name__}_{i}"
            results[name] = exporter.test_connection()

        return results


# =============================================================================
# Singleton Instance
# =============================================================================

siem_manager = SIEMManager()


# =============================================================================
# Helper Functions
# =============================================================================


def export_to_siem(entry) -> bool:
    """
    Export an audit entry to SIEM platforms.

    Args:
        entry: SysLogEntry instance

    Returns:
        bool: True if exported successfully
    """
    return siem_manager.export_audit_entry(entry)


def create_siem_event(
    event_name: str,
    tenant_id: str,
    correlation_id: str | None = None,
    severity: str = "medium",
    actor_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    action: str | None = None,
    status: str | None = None,
    payload: dict[str, Any] | None = None,
) -> SIEMEvent:
    """
    Create a SIEMEvent from components.

    Args:
        event_name: Event name
        tenant_id: Tenant identifier
        correlation_id: Correlation ID (generated if not provided)
        severity: Severity level (info, low, medium, high, critical)
        actor_id: Actor identifier
        resource_type: Resource type
        resource_id: Resource identifier
        action: Action description
        status: Status (success, failure, blocked, error)
        payload: Additional payload

    Returns:
        SIEMEvent instance
    """
    return SIEMEvent(
        event_id=str(uuid4()),
        correlation_id=correlation_id or str(uuid4()),
        timestamp=timezone.now(),
        event_name=event_name,
        severity=SEVERITY_MAP.get(severity, SIEMSeverity.MEDIUM),
        tenant_id=tenant_id,
        actor_id=actor_id,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        status=status,
        payload=payload or {},
    )
