# Intake verisimilitude samples (FieldSim v2 primitives)

So what: realistic, human-readable job intakes that correspond to the same underlying
structured primitives used by FieldSim v2 (budget, schedule, licensing/insurance, etc.).

Artifacts:
- `intake_samples.json`: structured sample texts + latent primitives

## city008_w00_easy_c000 (easy: Furniture assembly)

- budget_true: $168.0
- complexity_true: 0.923
- weirdness_true: 0.085
- requires_license_true: False
- requires_insurance_true: False

### Standard form
```
Standard intake form submission
City: San Francisco
Service: Furniture assembly
Budget (reported): $150
Timeline / availability:
- not provided (free-text only)
Requirements:
- licensed required? None
- insured required? None
Stated priorities (coarse):
- schedule_flexibility: high
- speed_urgency: medium
- price_sensitivity: low
Notes box: (optional)
- (left blank)
```

### AI chat
```
Agent: Hi — I’ll ask a few quick questions to write a clear request.
User: I’m in San Francisco. I need help with furniture assembly.
Agent: When do you want this done?
User: this week.
Agent: What budget range are you aiming for?
User: Around $130.
User: I’m somewhat flexible on price.
Agent: Any constraints like licensing/insurance requirements?
User: licensed required? no; insured required? no.
Agent: I heard: licensed=yes, insured=no.
Agent: What times work for you?
User: day 1 morning, day 1 afternoon, day 2 evening, (more).
Agent: Captured availability: day 1 morning, day 1 afternoon, day 2 evening, (more).
Agent: Anything else I should include?
User: No need for many updates.
Agent: Great — I’ll send this as a structured request.
```

## city008_w00_easy_c001 (easy: Faucet repair / replacement)

- budget_true: $174.24
- complexity_true: 0.903
- weirdness_true: 0.144
- requires_license_true: False
- requires_insurance_true: False

### Standard form
```
Standard intake form submission
City: San Francisco
Service: Faucet repair / replacement
Budget (reported): $150
Timeline / availability:
- not provided (free-text only)
Requirements:
- licensed required? None
- insured required? None
Stated priorities (coarse):
- weirdness_tolerance: high
- speed_urgency: medium
- price_sensitivity: low
Notes box: (optional)
- (left blank)
```

### AI chat
```
Agent: Hi — I’ll ask a few quick questions to write a clear request.
User: I’m in San Francisco. I need help with faucet repair / replacement.
Agent: When do you want this done?
User: ASAP.
Agent: What budget range are you aiming for?
User: Around $140.
User: I’m somewhat flexible on price.
Agent: Any constraints like licensing/insurance requirements?
User: licensed required? no; insured required? no.
Agent: I heard: licensed=no, insured=no.
Agent: What times work for you?
User: day 1 evening, day 3 morning, day 3 evening, (more).
Agent: Captured availability: day 1 evening, day 3 morning, day 3 evening, (more).
Agent: Anything else I should include?
User: No need for many updates.
Agent: Great — I’ll send this as a structured request.
```

## city008_w00_easy_c002 (easy: House cleaning)

- budget_true: $225.33
- complexity_true: 0.957
- weirdness_true: 0.114
- requires_license_true: False
- requires_insurance_true: False

### Standard form
```
Standard intake form submission
City: San Francisco
Service: House cleaning
Budget (reported): $225
Timeline / availability:
- not provided (free-text only)
Requirements:
- licensed required? None
- insured required? None
Stated priorities (coarse):
- speed_urgency: high
- schedule_flexibility: medium
- quality_focus: low
Notes box: (optional)
- (left blank)
```

### AI chat
```
Agent: Hi — I’ll ask a few quick questions to write a clear request.
User: I’m in San Francisco. I need help with house cleaning.
Agent: When do you want this done?
User: ASAP.
Agent: What budget range are you aiming for?
User: Around $220.
User: I’m somewhat flexible on price.
Agent: Any constraints like licensing/insurance requirements?
User: licensed required? no; insured required? no.
Agent: I heard: licensed=no, insured=yes.
Agent: What times work for you?
User: day 1 evening, day 2 evening, day 5 afternoon, (more).
Agent: Captured availability: day 1 evening, day 4 morning, day 5 afternoon, (more).
Agent: Anything else I should include?
User: No need for many updates.
Agent: Great — I’ll send this as a structured request.
```

## city008_w00_easy_c003 (easy: House cleaning)

- budget_true: $115.82
- complexity_true: 1.01
- weirdness_true: 0.0
- requires_license_true: False
- requires_insurance_true: False

### Standard form
```
Standard intake form submission
City: San Francisco
Service: House cleaning
Budget (reported): $100
Timeline / availability:
- not provided (free-text only)
Requirements:
- licensed required? None
- insured required? None
Stated priorities (coarse):
- speed_urgency: high
- communication_fit: medium
- price_sensitivity: low
Notes box: (optional)
- (left blank)
```

### AI chat
```
Agent: Hi — I’ll ask a few quick questions to write a clear request.
User: I’m in San Francisco. I need help with house cleaning.
Agent: When do you want this done?
User: ASAP.
Agent: What budget range are you aiming for?
User: Around $100.
User: I’m somewhat flexible on price.
Agent: Any constraints like licensing/insurance requirements?
User: licensed required? no; insured required? no.
Agent: I heard: licensed=no, insured=no.
Agent: What times work for you?
User: day 2 evening, day 3 evening, day 4 morning, (more).
Agent: Captured availability: day 2 evening, day 3 evening, day 4 morning, (more).
Agent: Anything else I should include?
User: Please send updates.
Agent: Great — I’ll send this as a structured request.
```

## city008_w00_easy_c004 (easy: Faucet repair / replacement)

- budget_true: $246.19
- complexity_true: 1.013
- weirdness_true: 0.289
- requires_license_true: True
- requires_insurance_true: False

### Standard form
```
Standard intake form submission
City: San Francisco
Service: Faucet repair / replacement
Budget (reported): $200
Timeline / availability:
- not provided (free-text only)
Requirements:
- licensed required? None
- insured required? None
Stated priorities (coarse):
- speed_urgency: high
- communication_fit: medium
- weirdness_tolerance: low
Notes box: (optional)
- (left blank)
```

### AI chat
```
Agent: Hi — I’ll ask a few quick questions to write a clear request.
User: I’m in San Francisco. I need help with faucet repair / replacement.
Agent: When do you want this done?
User: ASAP.
Agent: What budget range are you aiming for?
User: Around $210.
User: I’m somewhat flexible on price.
Agent: Any constraints like licensing/insurance requirements?
User: licensed required? yes; insured required? no.
Agent: I heard: licensed=yes, insured=no.
Agent: What times work for you?
User: day 1 afternoon, day 1 evening, day 2 afternoon, (more).
Agent: Captured availability: day 1 afternoon, day 1 evening, day 2 afternoon, (more).
Agent: Anything else I should include?
User: Please send updates.
Agent: Great — I’ll send this as a structured request.
```

## city008_w00_easy_c005 (easy: Drywall patch + paint touch-up)

- budget_true: $241.23
- complexity_true: 1.175
- weirdness_true: 0.133
- requires_license_true: False
- requires_insurance_true: False

### Standard form
```
Standard intake form submission
City: San Francisco
Service: Drywall patch + paint touch-up
Budget (reported): $250
Timeline / availability:
- not provided (free-text only)
Requirements:
- licensed required? None
- insured required? None
Stated priorities (coarse):
- schedule_flexibility: high
- price_sensitivity: medium
- quality_focus: low
Notes box: (optional)
- (left blank)
```

### AI chat
```
Agent: Hi — I’ll ask a few quick questions to write a clear request.
User: I’m in San Francisco. I need help with drywall patch + paint touch-up.
Agent: When do you want this done?
User: flexible.
Agent: What budget range are you aiming for?
User: Around $260.
User: I’m somewhat flexible on price.
Agent: Any constraints like licensing/insurance requirements?
User: licensed required? no; insured required? no.
Agent: I heard: licensed=no, insured=no.
Agent: What times work for you?
User: day 1 morning, day 2 afternoon, day 2 evening, (more).
Agent: Captured availability: day 1 morning, day 1 afternoon, day 2 afternoon, (more).
Agent: Anything else I should include?
User: No need for many updates.
Agent: Great — I’ll send this as a structured request.
```

## city008_w00_hard_c000 (hard: Electrical panel / wiring work)

- budget_true: $1151.83
- complexity_true: 1.84
- weirdness_true: 0.502
- requires_license_true: False
- requires_insurance_true: False

### Standard form
```
Standard intake form submission
City: San Francisco
Service: Electrical panel / wiring work
Budget (reported): $1150
Timeline / availability:
- not provided (free-text only)
Requirements:
- licensed required? None
- insured required? None
Stated priorities (coarse):
- schedule_flexibility: high
- communication_fit: medium
- quality_focus: low
Notes box: (optional)
- (left blank)
```

### AI chat
```
Agent: Hi — I’ll ask a few quick questions to write a clear request.
User: I’m in San Francisco. I need help with electrical panel / wiring work.
Agent: When do you want this done?
User: ASAP.
Agent: What budget range are you aiming for?
User: Around $1120.
User: I’m somewhat flexible on price.
Agent: Any constraints like licensing/insurance requirements?
User: licensed required? no; insured required? no.
Agent: I heard: licensed=no, insured=no.
Agent: What times work for you?
User: day 3 afternoon, day 3 evening, day 4 evening, (more).
Agent: Captured availability: day 3 afternoon, day 3 evening, day 4 evening, (more).
Agent: Anything else I should include?
User: No need for many updates. This might be more involved than it sounds; please sanity-check scope.
Agent: Great — I’ll send this as a structured request.
```

## city008_w00_hard_c001 (hard: Custom tile work (bath/kitchen))

- budget_true: $1763.36
- complexity_true: 1.738
- weirdness_true: 0.479
- requires_license_true: False
- requires_insurance_true: False

### Standard form
```
Standard intake form submission
City: San Francisco
Service: Custom tile work (bath/kitchen)
Budget (reported): $1775
Timeline / availability:
- not provided (free-text only)
Requirements:
- licensed required? None
- insured required? None
Stated priorities (coarse):
- communication_fit: high
- schedule_flexibility: medium
- speed_urgency: low
Notes box: (optional)
- (left blank)
```

### AI chat
```
Agent: Hi — I’ll ask a few quick questions to write a clear request.
User: I’m in San Francisco. I need help with custom tile work (bath/kitchen).
Agent: When do you want this done?
User: flexible.
Agent: What budget range are you aiming for?
User: Around $1860.
User: I’m somewhat flexible on price.
Agent: Any constraints like licensing/insurance requirements?
User: licensed required? no; insured required? no.
Agent: I heard: licensed=no, insured=no.
Agent: What times work for you?
User: day 1 morning, day 1 afternoon, day 2 morning, (more).
Agent: Captured availability: day 1 morning, day 1 afternoon, day 2 afternoon, (more).
Agent: Anything else I should include?
User: Please send updates. This might be more involved than it sounds; please sanity-check scope.
Agent: Great — I’ll send this as a structured request.
```

## city008_w00_hard_c002 (hard: Asbestos testing / remediation)

- budget_true: $6173.85
- complexity_true: 1.859
- weirdness_true: 0.711
- requires_license_true: True
- requires_insurance_true: True

### Standard form
```
Standard intake form submission
City: San Francisco
Service: Asbestos testing / remediation
Budget (reported): $5250
Timeline / availability:
- not provided (free-text only)
Requirements:
- licensed required? None
- insured required? None
Stated priorities (coarse):
- price_sensitivity: high
- quality_focus: medium
- schedule_flexibility: low
Notes box: (optional)
- (left blank)
```

### AI chat
```
Agent: Hi — I’ll ask a few quick questions to write a clear request.
User: I’m in San Francisco. I need help with asbestos testing / remediation.
Agent: When do you want this done?
User: flexible.
Agent: What budget range are you aiming for?
User: Around $4940.
User: I’m budget-conscious.
Agent: Any constraints like licensing/insurance requirements?
User: licensed required? yes; insured required? yes.
Agent: I heard: licensed=yes, insured=yes.
Agent: What times work for you?
User: day 1 morning, day 2 afternoon, day 2 evening, (more).
Agent: Captured availability: day 1 morning, day 2 evening, day 3 morning, (more).
Agent: Anything else I should include?
User: No need for many updates. This might be more involved than it sounds; please sanity-check scope.
Agent: Great — I’ll send this as a structured request.
```

## city008_w00_hard_c003 (hard: Custom tile work (bath/kitchen))

- budget_true: $1138.36
- complexity_true: 1.308
- weirdness_true: 0.722
- requires_license_true: False
- requires_insurance_true: True

### Standard form
```
Standard intake form submission
City: San Francisco
Service: Custom tile work (bath/kitchen)
Budget (reported): $975
Timeline / availability:
- not provided (free-text only)
Requirements:
- licensed required? None
- insured required? True
Stated priorities (coarse):
- price_sensitivity: high
- weirdness_tolerance: medium
- schedule_flexibility: low
Notes box: (optional)
- (left blank)
```

### AI chat
```
Agent: Hi — I’ll ask a few quick questions to write a clear request.
User: I’m in San Francisco. I need help with custom tile work (bath/kitchen).
Agent: When do you want this done?
User: flexible.
Agent: What budget range are you aiming for?
User: Around $910.
User: I’m budget-conscious.
Agent: Any constraints like licensing/insurance requirements?
User: licensed required? no; insured required? yes.
Agent: I heard: licensed=no, insured=yes.
Agent: What times work for you?
User: day 2 morning, day 2 evening, day 3 afternoon, (more).
Agent: Captured availability: day 2 morning, day 2 evening, day 3 afternoon, (more).
Agent: Anything else I should include?
User: No need for many updates.
Agent: Great — I’ll send this as a structured request.
```

## city008_w00_hard_c004 (hard: Electrical panel / wiring work)

- budget_true: $1651.66
- complexity_true: 1.416
- weirdness_true: 0.276
- requires_license_true: True
- requires_insurance_true: False

### Standard form
```
Standard intake form submission
City: San Francisco
Service: Electrical panel / wiring work
Budget (reported): $1400
Timeline / availability:
- not provided (free-text only)
Requirements:
- licensed required? None
- insured required? None
Stated priorities (coarse):
- price_sensitivity: high
- communication_fit: medium
- speed_urgency: low
Notes box: (optional)
- (left blank)
```

### AI chat
```
Agent: Hi — I’ll ask a few quick questions to write a clear request.
User: I’m in San Francisco. I need help with electrical panel / wiring work.
Agent: When do you want this done?
User: flexible.
Agent: What budget range are you aiming for?
User: Around $1320.
User: I’m budget-conscious.
Agent: Any constraints like licensing/insurance requirements?
User: licensed required? yes; insured required? no.
Agent: I heard: licensed=yes, insured=no.
Agent: What times work for you?
User: day 1 morning, day 1 afternoon, day 2 morning, (more).
Agent: Captured availability: day 1 morning, day 1 afternoon, day 1 evening, (more).
Agent: Anything else I should include?
User: Please send updates.
Agent: Great — I’ll send this as a structured request.
```

## city008_w00_hard_c005 (hard: Asbestos testing / remediation)

- budget_true: $11394.37
- complexity_true: 2.372
- weirdness_true: 1.0
- requires_license_true: True
- requires_insurance_true: False

### Standard form
```
Standard intake form submission
City: San Francisco
Service: Asbestos testing / remediation
Budget (reported): $11400
Timeline / availability:
- not provided (free-text only)
Requirements:
- licensed required? True
- insured required? None
Stated priorities (coarse):
- price_sensitivity: high
- quality_focus: medium
- speed_urgency: low
Notes box: (optional)
- (left blank)
```

### AI chat
```
Agent: Hi — I’ll ask a few quick questions to write a clear request.
User: I’m in San Francisco. I need help with asbestos testing / remediation.
Agent: When do you want this done?
User: flexible.
Agent: What budget range are you aiming for?
User: Around $11520.
User: I’m budget-conscious.
Agent: Any constraints like licensing/insurance requirements?
User: licensed required? yes; insured required? no.
Agent: I heard: licensed=yes, insured=no.
Agent: What times work for you?
User: day 1 afternoon, day 4 evening, day 5 afternoon, (more).
Agent: Captured availability: day 1 afternoon, day 3 evening, day 4 evening, (more).
Agent: Anything else I should include?
User: No need for many updates. It’s an old place and there might be surprises once you open it up.
Agent: Great — I’ll send this as a structured request.
```

