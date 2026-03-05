# Hyperlocal Deal Discovery Platform
### A Two-Sided Marketplace Connecting Local Stores with Deal-Seeking Consumers

---

## Table of Contents

1. [Mission](#mission)
2. [Problem Statement](#problem-statement)
3. [The Solution](#the-solution)
4. [Platform Architecture](#platform-architecture)
5. [Three Data Pillars](#three-data-pillars)
6. [Merchant Side](#merchant-side)
7. [Consumer Side](#consumer-side)
8. [Gmail Promotions Layer](#gmail-promotions-layer)
9. [Email Content Extraction — Full Handling](#email-content-extraction--full-handling)
10. [The 5 Indexing Dimensions](#the-5-indexing-dimensions)
11. [Vector DB & Retrieval Strategy](#vector-db--retrieval-strategy)
12. [Search Modes](#search-modes)
13. [Dimension 1 — CLIP Visual Embedding: Design Walkthrough](#dimension-1--clip-visual-embedding-design-walkthrough)
14. [Tech Stack](#tech-stack)
15. [Competitive Landscape](#competitive-landscape)
16. [Go-To-Market Strategy](#go-to-market-strategy)
17. [Monetization](#monetization)
18. [VC Analysis](#vc-analysis)

---

## Mission

> *"Empower local stores with their first digital presence and connect deal-seeking consumers to real-time neighborhood offers within walking distance."*

---

## Problem Statement

Millions of mom-and-pop stores have no digital presence. Their best deals never travel beyond the handwritten sign in their window. Meanwhile, consumers within walking distance are driving to big-box retailers or browsing Amazon — completely unaware that their neighborhood store has a better deal today. Both sides lose.

---

## The Solution

A two-sided hyperlocal deal platform where:

- **Store owners** post a photo, a price, and an item name in under 60 seconds
- That deal **instantly surfaces** in the feed of every consumer within a defined radius
- No website needed. No inventory system. No technical knowledge. Just a phone and a deal worth sharing
- Every post automatically **generates a structured inventory record** for the store owner — giving them their first real-time product catalog with zero extra effort

---

## Platform Architecture

```
Three Data Sources
        │
        ├── Gmail Promotions (personal inbox)
        │       └── 5-min aggregator → personalization layer
        │
        ├── Brand / Retailer Feeds (public)
        │       └── 6-hr aggregator → structured product data
        │
        └── Nearby Local Stores (hyperlocal — PRIMARY WEDGE)
                └── live store posts → real-time feed
        │
        ▼
Text + Image + Metadata
        │
        ▼
RAG Pipeline (Encoders — 5 Dimensions)
        │
        ▼
RAG System (Decoder)
        │
        ▼
UI: Search · Personalize · Live Feeds
```

---

## Three Data Pillars

### Pillar 1 — Gmail Promotions (Personal)
- User opts in via Gmail OAuth (promotions tab only — never personal inbox)
- 5-minute aggregator for near real-time inbox deal surfacing
- Feeds the personalization layer — knows what brands user subscribes to
- Privacy scoped: only reads what user already subscribed to from brands

### Pillar 2 — Brand / Retailer Feeds (Public)
- Public promotional data from major retailers (Nike, Old Navy, etc.)
- 6-hour aggregation cycle for catalog and promotional updates
- Structured product and pricing feeds

### Pillar 3 — Hyperlocal Store Posts (Physical — Primary Wedge)
- Mom and pop stores, local groceries, specialty shops
- Store owner posts photo + caption = live deal listing
- Geofenced consumer feed within defined radius
- Auto-generates structured inventory from each post
- The dataset that cannot be replicated by any competitor

---

## Merchant Side

### Why Mom & Pop is the Right Wedge

Flipp and Google Shopping require corporate API integrations and enterprise sales cycles. Your merchant is **Maria who runs the corner carnicería** — zero digital presence, zero technical knowledge, but a smartphone and great weekly deals.

The ask is minimal:
- Create a store profile (name, address, category)
- Post a photo of the item on sale
- Add a caption with the price

That is it. In return, they reach every deal-seeking consumer within walking distance — something they have never had before.

### Merchant Onboarding Flow

```
Step 1: Download app → create store profile
        (name, address, category, GPS auto-captured)

Step 2: Post a deal
        (photo + caption + optional category tag)

Step 3: Deal goes live in nearby consumer feed

Step 4: Inventory table auto-populates from post
        (store owner sees clean dashboard of all active deals)
```

### Auto-Generated Inventory Table

Every post automatically creates a structured inventory record:

| Item | Category | Price | Deal Type | Posted | Status |
|------|----------|-------|-----------|--------|--------|
| Fresh Salmon | Seafood | $6.99/lb | Flash | Today | Active |
| Ribeye Steak | Meat | $8.99/lb | Weekly | Yesterday | Active |
| Mango | Produce | $0.99 ea | Standard | 3 days ago | Expired |

Store owners never touch a spreadsheet. They post naturally and wake up to an organized inventory dashboard.

### Merchant Value Stack

- **Layer 1** — Digital presence (they exist online for the first time)
- **Layer 2** — Neighborhood reach (deals find nearby customers)
- **Layer 3** — Inventory intelligence (posts become business data)

Each layer increases the cost of leaving the platform. That is how you build merchant retention without a contract.

---

## Consumer Side

### The User Experience

```
Open app
    ↓
Real-time feed of deals from stores within radius
    ↓
Each card shows: store name · item photo · price · distance · time posted
    ↓
Search: "fresh fish near me" → semantic match across all sources
    ↓
Personalization: boosts categories and brands user engages with most
```

### What Makes It Different

A consumer searching "cheap ribeye near me" gets results pulled from:
- Their own Gmail promotions (brand deals they subscribed to)
- Nearby store posts (Maria's carnicería posted ribeye this morning)
- Public brand feeds (Costco weekly circular)

Three sources. One coherent ranked response. No other platform does this.

---

## Gmail Promotions Layer

### Core User Experience

```
User types: "I want to buy shoes"
        ↓
System checks: user subscribes to Nike, Adidas, Foot Locker
        ↓
Pulls active promotions from those brands
        ↓
Returns:
  → Nike: Air Max 90 — 30% off, ends Friday
  → Adidas: Ultraboost — $50 off, ends Sunday
  → Foot Locker: Up to 40% off running shoes
```

User never opened their inbox. You surfaced value they already had but couldn't see.

### Architecture — 5 Layers

**Layer 1 — Gmail Auth & Scoped Access**
- OAuth 2.0, narrowest possible scope
- Only promotions tab — never Primary, Social, or Updates
- Trust message: *"We only read your Promotions folder"*

**Layer 2 — Brand Subscription Graph**
- 90-day historical scan on first connect
- Builds per-user brand graph: Nike (14 emails/mo), Adidas (8/mo), etc.
- Becomes the personalization backbone

**Layer 3 — Promotion Extraction Pipeline**
- LLM structured extraction from raw email content
- Outputs: brand, product, discount, expiry, promo_code, deal_url

**Layer 4 — Deal Query & Retrieval (RAG)**
- Intent classification → brand graph lookup → semantic search → LLM response

**Layer 5 — Freshness & Sync**
- On app open: check for new emails since last sync
- Background: every 5 minutes
- On query: real-time check before responding
- Auto-expire deals past expiry date

---

## Email Content Extraction — Full Handling

Promotional emails are among the most complex HTML documents on the internet. The pipeline handles all 10 content types:

| Type | Description | Handling |
|------|-------------|----------|
| Pure text | Deal info in HTML text | Standard DOM parse |
| Text baked in image | "30% OFF" as image file | OCR + Vision LLM |
| Tracked/expiring image URLs | CDN URLs that expire | Download + cache to S3 immediately |
| Text beside image | Product shot + price text | DOM block association |
| Text below image | Caption pattern | Image-caption pairing |
| Hidden preview text | display:none spans | Capture separately — often best deal summary |
| CSS background images | Background-image with HTML overlay | Text is real HTML, extract normally + OCR background |
| Plain text MIME part | text/plain version of email | Parse first — often cleanest source |
| Animated GIFs | Multiple frames with different deals | Extract all frames, OCR each |
| Nested table HTML | Legacy email layout tables | DOM traversal with visual reading order |

### Unified Extraction Pipeline

```
Raw Email (MIME)
      ↓
MIME Parser → plain text · HTML · image URLs
      ↓
HTML Pre-processor → content blocks · hidden text · tracking pixel removal
      ↓
Image Pipeline (parallel per image)
  Download + cache to S3
  GIF detection → frame extraction
  OCR → raw text fragments
  Vision LLM → structured description
      ↓
Content Block Assembly
  HTML text + OCR text + alt text + caption text merged per block
      ↓
LLM Extraction → structured deal record per block
      ↓
Validation → confidence scoring → dual output
      ↓
  ├── Inventory Table (merchant side)
  └── Vector DB + Consumer Feed (consumer side)
```

---

## The 5 Indexing Dimensions

Indexing happens **per image** inside each email. One email with 3 deal-relevant images produces 3 deal units, each with 5 dimensions indexed.

```
1 Gmail Account
    └── many Emails
            └── each Email has many Images
                        └── each Image → 5 dimensions → 1 deal unit
```

---

### Dimension 1 — Visual Embedding (CLIP)

The raw pixel-level semantic representation of the image. No language translation. Pure visual semantics.

**Example:** A Nike promotional image of an Air Max 90 on a white background. CLIP encodes the shape of the sole, the mesh texture, the colorway, the product angle — into a 512-dimensional vector. The visual content itself becomes the vector.

---

### Dimension 2 — Structured Visual Attributes

Explicit, nameable product characteristics visible in the image.

**Example:** Air Max 90 image yields:
- color_primary: "white"
- color_secondary: "wolf grey"
- silhouette: "low-top"
- material: "mesh and leather"
- style_category: "lifestyle sneaker"
- gender_target: "unisex"
- occasion: "casual everyday"

---

### Dimension 3 — Ad/Deal Visual Signals

Visual cues about the promotion itself, not the product.

**Example:** Email banner has a red "LIMITED" badge, crossed-out original price, bold "SALE" sticker:
- has_limited_badge: true
- has_crossed_out_price: true
- has_sale_tag: true
- has_countdown_timer: false
- background_mood: "urgent"
- deal_prominence: "hero"

---

### Dimension 4 — Contextual / Lifestyle Signals

What the image communicates about who this is for and when and where it fits.

**Example:** Shoe photographed on a city sidewalk, worn by young adult in streetwear, warm summer lighting:
- lifestyle: "streetwear / casual"
- season: "summer"
- setting: "urban"
- occasion: "everyday"
- demographic: "young adult"

---

### Dimension 5 — Text Typography Signals

All text baked into the image via OCR, organized by visual prominence.

**Example:** Banner image contains:
- headline: "FLASH SALE" (large, bold, red)
- subheadline: "Air Max 90"
- body: "30% OFF SELECTED STYLES"
- fine_print: "Ends Feb 25. Exclusions apply."
- cta: "SHOP NOW"
- urgency_level: "high"

---

### Dimension Coverage by Email Type

| Email Type | D1 CLIP | D2 Attributes | D3 Signals | D4 Lifestyle | D5 Typography |
|------------|---------|---------------|------------|--------------|---------------|
| Pure text | ✗ | ✗ | partial | ✗ | partial |
| Pure image | ✓ | ✓ | ✓ | ✓ | ✓ |
| Mixed | ✓ | ✓ | ✓ | ✓ | ✓ |
| Image + short text | ✓ | ✓ | ✓ | ✓ | ✓ |
| Multi-block email | ✓ per image | ✓ per image | ✓ per image | ✓ per image | ✓ per image |

---

## Vector DB & Retrieval Strategy

### Two Embeddings Per Deal Unit

```
clip_embedding      VECTOR(512)    # Dimension 1 — raw visual CLIP
unified_embedding   VECTOR(1536)   # Dimensions 2-5 merged into one text corpus
```

The unified corpus that feeds the text embedding:
```
weighted_typography_text    (headline repeated 3x for weight)
+ attribute_text            (color, material, silhouette flattened)
+ signal_text               (sale signals, urgency flattened)
+ context_text              (lifestyle, season, occasion flattened)
+ ocr_text                  (raw OCR from images)
+ visual_desc               (free-form Vision LLM description)
+ html text_content         (extracted HTML text)
```

### Full Retrieval Flow

```
User query
    ↓
Query parser (LLM) → search_mode + hard filters + cleaned semantic query
    ↓
Query vectorization → query_text_vector (1536) + query_clip_vector (512)
    ↓
Vector DB — two parallel searches
    ├── cosine similarity on unified_embedding (top 20)
    └── cosine similarity on clip_embedding (top 20)
    ↓
Deduplicate + re-rank
    text_similarity * 0.35
    + clip_similarity * 0.25
    + urgency_score  * 0.20
    + discount_score * 0.15
    + recency_score  * 0.05
    ↓
Context assembly — top 5-7 structured deal records
    ↓
LLM response generation — natural language answer with recommendations
    ↓
User sees ranked deals
```

### Why Pass to LLM After Retrieval

The vector DB finds the right candidates. The LLM understands the user's actual need.

| Vector DB Can | LLM Can |
|---------------|---------|
| Find relevant deals | Compare deals |
| Filter by brand/price | Explain which is better |
| Rank by similarity | Handle vague queries |
| — | Make recommendations |
| — | Infer intent ("going running tomorrow" → running shoe) |

Two LLM calls per query: one to understand the query, one to generate the response.

---

## Search Modes

### Mode 1 — Natural Language Intent

```
"I want to buy shoes"
"looking for something warm this winter"
"need a gift for my wife under $50"
    ↓
Intent classification → category extraction → vector search → LLM response
```

### Mode 2 — Brand Search

```
"Show me Nike deals"
"any Adidas promotions?"
    ↓
Hard filter on brand field → group by category → sort by discount DESC
```

Fast and precise. No vector search needed for the primary filter.

### Mode 3 — Hybrid Search

```
"Nike running shoes under $100"
"Adidas deal ending soon"
    ↓
Parse into: brand filter + category intent + price constraint
    ↓
Hard filters first → semantic rerank within filtered set → ranking formula
```

The combination of structured filtering + semantic search that makes RAG genuinely useful for shopping.

---

## Dimension 1 — CLIP Visual Embedding: Design Walkthrough

### Technology Choices

| Component | Tool |
|-----------|------|
| CLIP Model | openai/clip-vit-base-patch32 (HuggingFace) |
| Vector DB | Pinecone (production) / ChromaDB (dev) |
| Backend | FastAPI |
| Image Storage | AWS S3 |
| Primary DB | PostgreSQL |
| Queue | Celery + Redis |

### Database Design

**PostgreSQL — deal_units table** stores all structured fields plus image references and clip_indexed status flag.

**Pinecone index:**
- Dimensions: 512 (CLIP vit-base-patch32)
- Metric: cosine
- Namespace: per user_id (user data isolation)
- Metadata stored alongside vector for filtering without PostgreSQL round trips

### Processing Pipeline

```
Step 1: Image URL arrives from Gmail parse
Step 2: Validate URL is reachable
Step 3: Download image
Step 4: Cache image to S3 (before tracked URL expires)
Step 5: Preprocess for CLIP (resize 224x224, normalize)
Step 6: Run CLIP image encoder → 512-dim vector
Step 7: Write vector + metadata to Pinecone
Step 8: Update PostgreSQL (clip_indexed=TRUE, S3 URL, indexed_at)
```

### Retrieval Pipeline

```
Step 1: User query arrives
Step 2: CLIP text encoder on query → 512-dim vector
Step 3: Query Pinecone (filter: user_id, status=active, expiry>now)
Step 4: Pinecone returns (unit_id, similarity_score, metadata)
Step 5: Fetch full records from PostgreSQL for top results
Step 6: Re-rank by similarity + discount + urgency
Step 7: Pass top 5-7 to LLM for response generation
```

### API Endpoints

```
POST   /api/v1/index/image        # index single image
POST   /api/v1/index/batch        # queue batch indexing job
GET    /api/v1/index/status/{id}  # check batch job progress
POST   /api/v1/search/clip        # search by query
DELETE /api/v1/index/{unit_id}    # remove from index
```

### Folder Structure

```
dimension1/
    ├── api/
    │     ├── main.py
    │     └── routes/
    │           ├── index.py
    │           └── search.py
    ├── core/
    │     ├── clip_encoder.py
    │     ├── image_processor.py
    │     ├── s3_client.py
    │     ├── pinecone_client.py
    │     └── postgres_client.py
    ├── pipeline/
    │     ├── index_pipeline.py
    │     └── search_pipeline.py
    ├── workers/
    │     └── celery_worker.py
    ├── models/
    │     └── deal_unit.py
    ├── config.py
    └── requirements.txt
```

### Validation Test

```
Index:   5 Nike shoe images
         3 Adidas apparel images
         2 Gap clothing images

Query:   "running shoes"
Expect:  Nike + Adidas shoe results surface
         Gap clothing does not surface

Query:   "casual streetwear"
Expect:  Apparel results surface over running shoes
```

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Image Encoding | CLIP (openai/clip-vit-base-patch32) |
| OCR | Google Vision API / AWS Textract |
| Vision LLM | Claude API (structured output) |
| Text Extraction | Claude API (tool_use) |
| Vector DB | Pinecone (prod) / ChromaDB (dev) |
| Primary DB | PostgreSQL + PostGIS |
| Pipeline Orchestration | LangGraph |
| Job Queue | Celery + Redis |
| Image Storage | AWS S3 |
| Backend API | FastAPI |
| Mobile App | React Native |
| Authentication | Firebase Auth |
| GIF Processing | Pillow (PIL) |
| HTML Parsing | BeautifulSoup4 |
| MIME Parsing | Python email library |
| QR Code Decoding | pyzbar |

---

## Competitive Landscape

| Competitor | What They Do | What You Have That They Don't |
|------------|--------------|-------------------------------|
| Flipp | Digitizes chain store flyers | Mom & pop network, Gmail layer, multimodal RAG |
| Basket | Trip cost comparison | Real-time local posts, image-based search |
| Flashfood | Perishable discounts | Broader categories, merchant UGC |
| Google Shopping | Online product search | Hyperlocal physical store data |
| Honey | Coupon codes at checkout | Proactive discovery, local deals, personalization |

### Core Moats

1. Only platform giving mom & pop stores a free digital presence
2. User-generated deal content building a proprietary hyperlocal dataset
3. Multimodal RAG searching across image + text + structured inventory simultaneously
4. Gmail personalization layer no competitor has replicated
5. Geographic network effects — once you own a neighborhood feed, extremely hard to displace

---

## Go-To-Market Strategy

### Month 1-2
- Manually onboard 5-10 local grocery / specialty stores in one neighborhood
- Hand-hold first posts, verify inventory table populates correctly
- Recruit 100 local consumer users from same neighborhood
- Key question: does a consumer find a deal they wouldn't have found otherwise?

### Month 2-3
- Build consumer app with RAG-powered search and nearby feed
- Automate post → inventory extraction pipeline
- Measure: retention, deal redemption rate, store post frequency

### Month 4-6
- If retention holds, productize merchant upload flow
- Expand to 2-3 adjacent neighborhoods
- Build store owner analytics (reach, clicks, directions)

### Month 6-12
- City-level launch in one metro area
- Add Gmail promotions integration
- Explore performance-based monetization

---

## Monetization

### Phase 1 — Free (Growth)
Free for all merchants and consumers. Goal: build the network, own the hyperlocal deal graph.

### Phase 2 — Merchant Premium
- Analytics: which deals drove foot traffic
- Promoted placement: deal appears higher in consumer feed
- Performance-based: pay per redemption (zero risk for merchant)

### Phase 3 — Consumer Premium
- Deal alerts: notify me when X drops below $Y near me
- Gmail inbox integration (promotions layer)
- Trip planning: find best prices across nearby stores for a shopping list

---

## VC Analysis

### The One-Liner

> *"Flipp digitized the paper flyer for chain stores. We're giving every mom and pop store their first digital presence — a 60-second photo post that instantly reaches every deal-seeker within walking distance. Our multimodal RAG pipeline turns each post into a structured inventory record automatically, building the only real-time hyperlocal deal graph that no big tech platform has."*

### Why Now

- Smartphone penetration among small business owners is near universal
- Consumer appetite for local and value-driven shopping is accelerating
- Hyperlocal digital commerce for stores with zero online presence is almost entirely unsolved
- Multimodal AI has matured enough to make image-based extraction reliable at scale

### The Critical Path

The tech is buildable. The real question is merchant acquisition — specifically whether you can sign enough local stores in one neighborhood to make the consumer feed feel alive. That is the only metric that matters at seed stage.

---

*Document generated from design sessions covering platform architecture, Gmail promotions pipeline, multimodal email extraction, 5-dimension indexing strategy, vector DB retrieval design, and Dimension 1 CLIP implementation walkthrough.*