## Conversation

### Turn 1

**User**

> We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?

**Agent**

Before I shape the stack — what language are the calls in? That drives which spoken-language screen we use.

_No recommendations this turn (`recommendations: null`)._

_`end_of_conversation`: **false**_

### Turn 2

**User**

> English.

**Agent**

SVAR has four English variants in the catalog: US, UK, Australian, and Indian accent. The choice matters because the screen is calibrated for the accent your callers will hear. Which fits your operation?

_No recommendations this turn (`recommendations: null`)._

_`end_of_conversation`: **false**_

### Turn 3

**User**

> US.

**Agent**

For high-volume entry-level contact centre screening in English US, the right stack layers a spoken language screen, a simulation, and a behavioural fit component:

| # | Name | Test Type | Keys | Duration | Languages | URL |
|---|------|-----------|------|----------|-----------|-----|
| 1 | SVAR Spoken English (US) (New) | K | Simulations | - | English (USA) | <https://www.shl.com/products/product-catalog/view/svar-spoken-english-us-new/> |
| 2 | Contact Center Call Simulation (New) | S | Simulations | 15 minutes | English (USA) | <https://www.shl.com/products/product-catalog/view/contact-center-call-simulation-new/> |
| 3 | Entry Level Customer Serv - Retail & Contact Center | P,C | Personality & Behavior, Competencies | 19 minutes | Latin American Spanish, German, French, Chinese Simplified _(+10 more)_ | <https://www.shl.com/products/product-catalog/view/entry-level-customer-serv-retail-and-contact-center/> |
| 4 | Customer Service Phone Simulation | B,S | Biodata & Situational Judgment, Simulations | 20 minutes | French (Canada), Portuguese (Brazil), Dutch, Italian _(+7 more)_ | <https://www.shl.com/products/product-catalog/view/customer-service-phone-simulation/> |

_`end_of_conversation`: **false**_

### Turn 4

**User**

> Is the Contact Center Call Simulation different from the Customer Service Phone Simulation?

**Agent**

Yes — distinct products. The Customer Service Phone Simulation is an older bundled solution (B, P, S) combining personality, behaviour, and simulation in one package. The Contact Center Call Simulation (New) is a standalone, newer simulation focused purely on the in-call interaction. Many clients use the new simulation for volume screening and the older solution for finalist-stage depth.

_No recommendations this turn (`recommendations: null`)._

_`end_of_conversation`: **false**_

### Turn 5

**User**

> Perfect — new simulation for volume, old solution for finalists. Confirmed.

**Agent**

Good two-stage design.

| # | Name | Test Type | Keys | Duration | Languages | URL |
|---|------|-----------|------|----------|-----------|-----|
| 1 | SVAR Spoken English (US) (New) | K | Simulations | - | English (USA) | <https://www.shl.com/products/product-catalog/view/svar-spoken-english-us-new/> |
| 2 | Contact Center Call Simulation (New) | S | Simulations | 15 minutes | English (USA) | <https://www.shl.com/products/product-catalog/view/contact-center-call-simulation-new/> |
| 3 | Entry Level Customer Serv - Retail & Contact Center | P,C | Personality & Behavior, Competencies | 19 minutes | Latin American Spanish, German, French, Chinese Simplified _(+10 more)_ | <https://www.shl.com/products/product-catalog/view/entry-level-customer-serv-retail-and-contact-center/> |
| 4 | Customer Service Phone Simulation | B,S | Biodata & Situational Judgment, Simulations | 20 minutes | French (Canada), Portuguese (Brazil), Dutch, Italian _(+7 more)_ | <https://www.shl.com/products/product-catalog/view/customer-service-phone-simulation/> |

_`end_of_conversation`: **true**_
