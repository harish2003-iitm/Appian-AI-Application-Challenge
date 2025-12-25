"""
Layer 2: Ontology Layer - Digital Twin of the Organization
- Defines business objects: Claims, Policies, Patients, Providers
- Enables relational queries across objects
- Acts as semantic layer on top of raw data
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json


class ClaimStatus(Enum):
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    DENIED = "denied"
    PENDING_INFO = "pending_info"
    APPEALED = "appealed"
    CLOSED = "closed"


class PolicyType(Enum):
    FLOOD = "flood"
    AUTO = "auto"
    HEALTH = "health"
    PROPERTY = "property"
    LIABILITY = "liability"
    WORKERS_COMP = "workers_comp"


@dataclass
class OntologyObject:
    """Base class for all ontology objects"""
    id: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.__class__.__name__,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }

    def to_searchable_text(self) -> str:
        """Convert to text for indexing"""
        raise NotImplementedError


@dataclass
class Policy(OntologyObject):
    """Insurance Policy object"""
    policy_number: str = ""
    policy_type: PolicyType = PolicyType.FLOOD
    holder_name: str = ""
    effective_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None
    premium: float = 0.0
    deductible: float = 0.0
    coverage_limit: float = 0.0
    state: str = ""
    coverages: List[str] = field(default_factory=list)
    endorsements: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)

    def to_searchable_text(self) -> str:
        return f"""Policy: {self.policy_number}
Type: {self.policy_type.value}
Holder: {self.holder_name}
State: {self.state}
Premium: ${self.premium:,.2f}
Deductible: ${self.deductible:,.2f}
Coverage Limit: ${self.coverage_limit:,.2f}
Coverages: {', '.join(self.coverages)}
Endorsements: {', '.join(self.endorsements)}
Exclusions: {', '.join(self.exclusions)}"""


@dataclass
class Claim(OntologyObject):
    """Insurance Claim object"""
    claim_number: str = ""
    policy_id: str = ""  # Links to Policy
    claimant_name: str = ""
    date_of_loss: Optional[datetime] = None
    date_reported: Optional[datetime] = None
    status: ClaimStatus = ClaimStatus.SUBMITTED
    claim_type: str = ""
    description: str = ""
    amount_claimed: float = 0.0
    amount_approved: float = 0.0
    denial_reason: str = ""
    adjuster_id: str = ""
    provider_ids: List[str] = field(default_factory=list)  # Links to Providers
    documents: List[str] = field(default_factory=list)

    def to_searchable_text(self) -> str:
        return f"""Claim: {self.claim_number}
Policy: {self.policy_id}
Claimant: {self.claimant_name}
Type: {self.claim_type}
Status: {self.status.value}
Description: {self.description}
Amount Claimed: ${self.amount_claimed:,.2f}
Amount Approved: ${self.amount_approved:,.2f}
Denial Reason: {self.denial_reason}"""


@dataclass
class Provider(OntologyObject):
    """Healthcare/Service Provider object"""
    npi: str = ""  # National Provider Identifier
    name: str = ""
    provider_type: str = ""  # Hospital, Physician, etc.
    specialty: str = ""
    address: str = ""
    state: str = ""
    in_network: bool = True
    tax_id: str = ""
    claims_count: int = 0
    total_billed: float = 0.0

    def to_searchable_text(self) -> str:
        return f"""Provider: {self.name}
NPI: {self.npi}
Type: {self.provider_type}
Specialty: {self.specialty}
State: {self.state}
In Network: {'Yes' if self.in_network else 'No'}
Total Claims: {self.claims_count}
Total Billed: ${self.total_billed:,.2f}"""


@dataclass
class Patient(OntologyObject):
    """Patient/Member object"""
    member_id: str = ""
    name: str = ""
    date_of_birth: Optional[datetime] = None
    policy_ids: List[str] = field(default_factory=list)
    primary_care_provider: str = ""
    conditions: List[str] = field(default_factory=list)
    claims_history: List[str] = field(default_factory=list)

    def to_searchable_text(self) -> str:
        return f"""Patient/Member: {self.name}
Member ID: {self.member_id}
Policies: {', '.join(self.policy_ids)}
PCP: {self.primary_care_provider}
Conditions: {', '.join(self.conditions)}"""


@dataclass
class Regulation(OntologyObject):
    """Regulatory/Compliance object"""
    regulation_id: str = ""
    title: str = ""
    authority: str = ""  # CMS, State DOI, etc.
    effective_date: Optional[datetime] = None
    applies_to: List[str] = field(default_factory=list)  # Policy types
    states: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    penalties: str = ""

    def to_searchable_text(self) -> str:
        return f"""Regulation: {self.regulation_id}
Title: {self.title}
Authority: {self.authority}
Applies To: {', '.join(self.applies_to)}
States: {', '.join(self.states)}
Requirements: {chr(10).join('- ' + r for r in self.requirements)}
Penalties: {self.penalties}"""


class OntologyStore:
    """
    Central store for all ontology objects.
    Enables relational queries across objects.
    """

    def __init__(self):
        self.policies: Dict[str, Policy] = {}
        self.claims: Dict[str, Claim] = {}
        self.providers: Dict[str, Provider] = {}
        self.patients: Dict[str, Patient] = {}
        self.regulations: Dict[str, Regulation] = {}

        # Indexes for fast lookups
        self._policy_by_number: Dict[str, str] = {}
        self._claim_by_number: Dict[str, str] = {}
        self._provider_by_npi: Dict[str, str] = {}
        self._claims_by_policy: Dict[str, List[str]] = {}
        self._claims_by_provider: Dict[str, List[str]] = {}

    def add_policy(self, policy: Policy) -> str:
        self.policies[policy.id] = policy
        self._policy_by_number[policy.policy_number] = policy.id
        return policy.id

    def add_claim(self, claim: Claim) -> str:
        self.claims[claim.id] = claim
        self._claim_by_number[claim.claim_number] = claim.id

        # Index by policy
        if claim.policy_id:
            if claim.policy_id not in self._claims_by_policy:
                self._claims_by_policy[claim.policy_id] = []
            self._claims_by_policy[claim.policy_id].append(claim.id)

        # Index by providers
        for provider_id in claim.provider_ids:
            if provider_id not in self._claims_by_provider:
                self._claims_by_provider[provider_id] = []
            self._claims_by_provider[provider_id].append(claim.id)

        return claim.id

    def add_provider(self, provider: Provider) -> str:
        self.providers[provider.id] = provider
        self._provider_by_npi[provider.npi] = provider.id
        return provider.id

    def add_patient(self, patient: Patient) -> str:
        self.patients[patient.id] = patient
        return patient.id

    def add_regulation(self, regulation: Regulation) -> str:
        self.regulations[regulation.id] = regulation
        return regulation.id

    # Relational Queries

    def get_claims_for_policy(self, policy_id: str) -> List[Claim]:
        """Get all claims linked to a policy"""
        claim_ids = self._claims_by_policy.get(policy_id, [])
        return [self.claims[cid] for cid in claim_ids if cid in self.claims]

    def get_claims_for_provider(self, provider_id: str) -> List[Claim]:
        """Get all claims linked to a provider"""
        claim_ids = self._claims_by_provider.get(provider_id, [])
        return [self.claims[cid] for cid in claim_ids if cid in self.claims]

    def get_claims_by_provider_above_threshold(self, provider_id: str, threshold: float) -> List[Claim]:
        """Show all Claims linked to Provider where total value exceeds threshold"""
        claims = self.get_claims_for_provider(provider_id)
        return [c for c in claims if c.amount_claimed > threshold]

    def get_policy_for_claim(self, claim_id: str) -> Optional[Policy]:
        """Get the policy associated with a claim"""
        claim = self.claims.get(claim_id)
        if claim and claim.policy_id:
            return self.policies.get(claim.policy_id)
        return None

    def get_providers_for_claim(self, claim_id: str) -> List[Provider]:
        """Get all providers associated with a claim"""
        claim = self.claims.get(claim_id)
        if not claim:
            return []
        return [self.providers[pid] for pid in claim.provider_ids if pid in self.providers]

    def get_regulations_for_policy_type(self, policy_type: PolicyType, state: str = None) -> List[Regulation]:
        """Get applicable regulations for a policy type"""
        results = []
        for reg in self.regulations.values():
            if policy_type.value in reg.applies_to:
                if state is None or state in reg.states or not reg.states:
                    results.append(reg)
        return results

    def get_high_value_claims_by_provider(self, threshold: float = 50000) -> Dict[str, List[Claim]]:
        """Find providers with high-value claims (fraud detection use case)"""
        results = {}
        for provider_id, claim_ids in self._claims_by_provider.items():
            high_value = []
            for cid in claim_ids:
                claim = self.claims.get(cid)
                if claim and claim.amount_claimed > threshold:
                    high_value.append(claim)
            if high_value:
                results[provider_id] = high_value
        return results

    # Natural Language Query Interface

    def query(self, question: str) -> Dict:
        """
        Natural language query interface.
        Parses question and returns relevant ontology objects.
        """
        question_lower = question.lower()
        results = {"objects": [], "query_type": "", "summary": ""}

        # Pattern matching for common queries
        if "claims" in question_lower and "provider" in question_lower:
            # Extract provider reference
            import re
            provider_match = re.search(r'provider\s+(\w+)', question_lower)
            threshold_match = re.search(r'exceed[s]?\s+\$?([\d,]+)', question_lower)

            if provider_match:
                provider_name = provider_match.group(1)
                # Find provider
                for p in self.providers.values():
                    if provider_name in p.name.lower():
                        threshold = float(threshold_match.group(1).replace(',', '')) if threshold_match else 0
                        claims = self.get_claims_by_provider_above_threshold(p.id, threshold)
                        results["objects"] = [c.to_dict() for c in claims]
                        results["query_type"] = "claims_by_provider"
                        results["summary"] = f"Found {len(claims)} claims for {p.name} exceeding ${threshold:,.2f}"
                        break

        elif "policy" in question_lower:
            # Extract policy number
            import re
            policy_match = re.search(r'policy\s*#?\s*(\w+)', question_lower)
            if policy_match:
                policy_num = policy_match.group(1)
                policy_id = self._policy_by_number.get(policy_num)
                if policy_id:
                    policy = self.policies[policy_id]
                    results["objects"] = [policy.to_dict()]
                    results["query_type"] = "policy_lookup"
                    results["summary"] = f"Found policy {policy.policy_number}"

        elif "denied" in question_lower or "denial" in question_lower:
            denied_claims = [c for c in self.claims.values() if c.status == ClaimStatus.DENIED]
            results["objects"] = [c.to_dict() for c in denied_claims]
            results["query_type"] = "denied_claims"
            results["summary"] = f"Found {len(denied_claims)} denied claims"

        return results

    # Serialization

    def to_searchable_documents(self) -> List[Dict]:
        """Convert all objects to searchable documents"""
        documents = []

        for policy in self.policies.values():
            documents.append({
                "id": policy.id,
                "type": "policy",
                "content": policy.to_searchable_text(),
                "metadata": policy.to_dict()
            })

        for claim in self.claims.values():
            documents.append({
                "id": claim.id,
                "type": "claim",
                "content": claim.to_searchable_text(),
                "metadata": claim.to_dict()
            })

        for provider in self.providers.values():
            documents.append({
                "id": provider.id,
                "type": "provider",
                "content": provider.to_searchable_text(),
                "metadata": provider.to_dict()
            })

        for patient in self.patients.values():
            documents.append({
                "id": patient.id,
                "type": "patient",
                "content": patient.to_searchable_text(),
                "metadata": patient.to_dict()
            })

        for regulation in self.regulations.values():
            documents.append({
                "id": regulation.id,
                "type": "regulation",
                "content": regulation.to_searchable_text(),
                "metadata": regulation.to_dict()
            })

        return documents

    def save_to_json(self, filepath: str):
        """Save ontology to JSON file"""
        data = {
            "policies": {k: v.to_dict() for k, v in self.policies.items()},
            "claims": {k: v.to_dict() for k, v in self.claims.items()},
            "providers": {k: v.to_dict() for k, v in self.providers.items()},
            "patients": {k: v.to_dict() for k, v in self.patients.items()},
            "regulations": {k: v.to_dict() for k, v in self.regulations.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def get_stats(self) -> Dict:
        """Get ontology statistics"""
        return {
            "policies": len(self.policies),
            "claims": len(self.claims),
            "providers": len(self.providers),
            "patients": len(self.patients),
            "regulations": len(self.regulations),
            "total_objects": sum([
                len(self.policies), len(self.claims), len(self.providers),
                len(self.patients), len(self.regulations)
            ])
        }
