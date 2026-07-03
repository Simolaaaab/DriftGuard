# Cross-LLM anomaly catalogue (paper appendix)

Six anomaly classes extracted from the multi-LLM bench (drift_anchored + B + K=20, seed 2025). Top 6 cases per class, ranked by severity.

## Counts per class (all providers pooled)

- **confident_wrong**: 310 instances
- **cross_model_split**: 243 instances
- **lucky_halluc**: 49 instances
- **groundless_phish**: 0 instances
- **inverse_conservatism**: 392 instances
- **format_failure**: 0 instances

---

## confident_wrong  (310 total)

### 1. DeepSeek-V4-Flash — sample 1430

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: ``  |  **GT**: 1  |  **correct**: 0
- **Confidence**: 0.99  |  **Trust**: 0.87  |  **n_found/n_not/n_contr**: 0/13/1
- **Reasoning**: The URL domain is imdb.com, a top-250 Tranco site with a 30-year domain age and extensive Wayback history, strongly indicating legitimacy. The HTML content shows standard IMDb meta tags, Open Graph properties, and Amazon analytics scripts, with no password fields, login forms, credential-harvesting JavaScript, hidden iframes, or brand impersonation. The page title 'Private list' and description 'The creator of this list has not enabled public viewing' match a legitimate IMDb feature for private 


### 2. DeepSeek-V4-Flash — sample 2299

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: ``  |  **GT**: 1  |  **correct**: 0
- **Confidence**: 0.99  |  **Trust**: 0.87  |  **n_found/n_not/n_contr**: 5/8/1
- **Reasoning**: The URL domain is youtube.com, a mainstream site ranked #13,105 by Tranco, and the HTML content is the legitimate YouTube homepage with standard YouTube scripts, favicons, meta tags, and canonical links. No password fields, login forms, credential-harvesting JavaScript, hidden iframes, or brand impersonation are present. The OSINT dossier confirms the domain is well-established and hosted on a reputable ASN, further supporting benign classification.


### 3. DeepSeek-V4-Flash — sample 745

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: ``  |  **GT**: 1  |  **correct**: 0
- **Confidence**: 0.99  |  **Trust**: 0.95  |  **n_found/n_not/n_contr**: 0/10/0
- **Reasoning**: The URL is a legitimate Blogger blogspot.com subdomain (cibopassionemente.blogspot.com) with a 19-year-old domain, Tranco rank #48,988, and RDAP registration. The HTML content is a standard Blogger template with no login forms, password fields, credential-harvesting scripts, hidden iframes, or brand impersonation. The page is a food blog in Italian, consistent with its title and navigation links. No phishing indicators were found.


### 4. GPT-5.4-mini — sample 1430

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: ``  |  **GT**: 1  |  **correct**: 0
- **Confidence**: 0.99  |  **Trust**: 0.87  |  **n_found/n_not/n_contr**: 4/5/1
- **Reasoning**: The URL/context points to IMDb, a highly established domain with strong OSINT support for legitimacy: Tranco top-1k ranking, 30.4-year domain age, and extensive Wayback history. In the HTML, I found an IMDb page titled 'Private list' with standard IMDb/OpenGraph metadata and normal Amazon/IMDb scripts; I did not find password fields, login forms, credential-harvest JavaScript, hidden iframes, fake CAPTCHA, redirects, or brand-impersonation cues beyond the genuine IMDb branding. The page content 


### 5. GPT-5.4-mini — sample 1520

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: ``  |  **GT**: 1  |  **correct**: 0
- **Confidence**: 0.99  |  **Trust**: 0.86  |  **n_found/n_not/n_contr**: 5/7/1
- **Reasoning**: The URL/domain context points to a long-established, highly popular site: Tranco #244, 30.4 years old, with 1000 Wayback snapshots and registrar/RDAP presence, all of which strongly argue against a throwaway phishing domain. The HTML content is consistent with an IMDb page titled "Private list" and includes IMDb/Open Graph metadata, Amazon/IMDb analytics scripts, and no login or payment capture elements; I checked for password fields, credential-harvest forms, hidden iframes, fake CAPTCHA, redir


### 6. GPT-5.4-mini — sample 568

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: ``  |  **GT**: 1  |  **correct**: 0
- **Confidence**: 0.99  |  **Trust**: 0.87  |  **n_found/n_not/n_contr**: 4/6/1
- **Reasoning**: The URL/context points to IMDb, an established top-1k domain with a 30.4-year domain age, 1000+ Wayback snapshots, and long-lived RDAP/ASN signals, which are strong legitimacy indicators. The HTML matches a normal IMDb page for a private list: title/OG tags say “Private list,” and the description says the creator has not enabled public viewing; I did not find password fields, login forms, credential-harvest JavaScript, hidden iframes, fake CAPTCHA, redirects, or brand-impersonation beyond the le


---

## lucky_halluc  (49 total)

### 1. Mistral-Large-3 — sample 1679

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: `1`  |  **GT**: 1  |  **correct**: 1
- **Confidence**: 0.95  |  **Trust**: 0.40  |  **n_found/n_not/n_contr**: 6/10/4
- **Reasoning**: The domain 'gestaodecadastrosatualizar.blogspot.com' is a free Blogspot subdomain, which is commonly abused for phishing due to its low barrier to entry and perceived legitimacy. The page title 'gestao de cadastros' (Portuguese for 'management of registrations') and the 404-like message 'A página que você está procurando neste blog não existe' suggest a lure to redirect or harvest credentials, despite no explicit login form being present. The OSINT dossier shows an established domain (blogspot.c


### 2. Mistral-Large-3 — sample 869

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: `1`  |  **GT**: 1  |  **correct**: 1
- **Confidence**: 0.98  |  **Trust**: 0.46  |  **n_found/n_not/n_contr**: 9/5/4
- **Reasoning**: The page impersonates Roblox with a 'Get Robux' offer, a classic credential-harvesting lure. It contains a username input field, fake verification steps, and a hidden iframe modal likely used for task-based scams or redirects to credential harvesters. The domain is legitimate (Tranco #5,757, 18.9 years old), but the HTML content is clearly phishing: it uses CloudFront-hosted JS with affiliate IDs, fake payout testimonials, and a 'verify now' button that triggers obfuscated behavior. The OSINT do


### 3. DeepSeek-V4-Flash — sample 933

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: `1`  |  **GT**: 1  |  **correct**: 1
- **Confidence**: 0.92  |  **Trust**: 0.48  |  **n_found/n_not/n_contr**: 7/7/3
- **Reasoning**: The page is hosted on a blogspot.com subdomain (trezorrloggin.blogspot.com) which is a free hosting platform, and the domain name contains a typo-squatting variant of 'trezor' with an extra 'r' and 'loggin' instead of 'login'. The HTML content impersonates the Trezor brand, includes multiple references to 'Trezor login' and 'Trezor Suite', and contains a large image of the Trezor start page, but lacks any actual login form or credential-harvesting mechanism visible in the cleaned HTML. However, 


### 4. GPT-5.4 — sample 1225

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: `1`  |  **GT**: 1  |  **correct**: 1
- **Confidence**: 0.96  |  **Trust**: 0.61  |  **n_found/n_not/n_contr**: 12/10/3
- **Reasoning**: The HTML contains a scripted redirect after 100ms to //playgwent-kouryaku.xyz/click2131287361236.php with an encoded external destination, which is a strong malicious-redirect indicator. The visible content is a flashing get-rich-quick lure ('Deposit 2000 USD -> GET 200 000 USD X100') plus hundreds of autogenerated numeric links on the same suspicious .xyz domain, consistent with spam/scam doorway pages rather than a legitimate site. I checked for login forms and password harvesting and did not 


### 5. GPT-5.4 — sample 1543

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: `1`  |  **GT**: 1  |  **correct**: 1
- **Confidence**: 0.93  |  **Trust**: 0.57  |  **n_found/n_not/n_contr**: 8/10/3
- **Reasoning**: The HTML shows a shopping site on stanleyjeudy.shop but repeatedly references a different domain, clean-commerce.com, in the canonical URL, Open Graph URL, and schema WebPage URL, which is a strong indicator of copied or mismatched site content. The page title and description claim 'Sanctuary Clothing' while the site name is 'Stanley Jeudy,' creating explicit brand/content inconsistency typical of fake storefront impersonation. The content appears to be a WooCommerce WordPress shop template rath


### 6. GPT-5.4 — sample 611

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: `1`  |  **GT**: 1  |  **correct**: 1
- **Confidence**: 0.88  |  **Trust**: 0.57  |  **n_found/n_not/n_contr**: 10/10/3
- **Reasoning**: The HTML shows an e-commerce storefront on the domain holgersasse.shop, but the page title, description, logo image, product catalog, and navigation all advertise the luxury fashion brand Safiyaa rather than 'Holger Sasse', which is a concrete brand/domain mismatch indicative of impersonation. The site uses a low-reputation .shop domain not in Tranco, with limited CT history and only a short recent certificate history, which supports suspicion. I checked for credential harvesting elements and di


---

## inverse_conservatism  (392 total)

### 1. DeepSeek-V4-Flash — sample 755

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: ``  |  **GT**: 1  |  **correct**: 0
- **Confidence**: 0.95  |  **Trust**: 0.83  |  **n_found/n_not/n_contr**: 0/16/5
- **Reasoning**: The URL is a standard blogspot.com subdomain (lilbedwordweave.blogspot.com), which is a legitimate Google-owned blogging platform. The HTML content is a typical Blogger template with blog posts, comments, and standard Google APIs (jQuery, Blogger widgets, Google Fonts, Font Awesome). There are no password fields, login forms, credential-harvesting scripts, hidden iframes, brand impersonation, fake CAPTCHAs, redirects, obfuscated content, wallet drainers, fake updates, or fake stores. The OSINT d


### 2. GPT-5.4 — sample 2511

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: ``  |  **GT**: 0  |  **correct**: 1
- **Confidence**: 0.98  |  **Trust**: 0.83  |  **n_found/n_not/n_contr**: 10/12/5
- **Reasoning**: The HTML is consistent with a legitimate Los Angeles Times article page: it contains article Open Graph and Twitter metadata, og:site_name='Los Angeles Times', og:url on https://www.latimes.com/..., and numerous normal analytics/advertising integrations used by major news sites. I found no login form, no password or payment fields, no wallet-connect prompt, no fake CAPTCHA, no browser-update lure, and no credential-posting JavaScript. The context strongly supports legitimacy as well: the site is


### 3. GPT-5.4 — sample 3192

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: ``  |  **GT**: 0  |  **correct**: 1
- **Confidence**: 0.98  |  **Trust**: 0.84  |  **n_found/n_not/n_contr**: 14/15/5
- **Reasoning**: The HTML content is consistent with a legitimate Delta-owned business site: multiple explicit references to skymilesforbusiness.delta.com, pro.delta.com, signin.delta.com, okta.delta.com, and Salesforce-hosted Delta properties appear throughout the CSP and asset allowlists. I checked for common phishing traits and did not find a credential form, password fields, fake CAPTCHA gates, hidden iframes, obfuscated credential-harvesting JavaScript, crypto wallet prompts, fake browser update messaging, 


### 4. GPT-5.4 — sample 3335

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: ``  |  **GT**: 0  |  **correct**: 1
- **Confidence**: 0.97  |  **Trust**: 0.84  |  **n_found/n_not/n_contr**: 9/17/5
- **Reasoning**: The HTML contains numerous standard analytics, advertising, and performance-monitoring scripts, and explicitly references the legitimate domain "www.marriott.com" and "domain=marriott.com" in Facebook Pixel and Dynatrace configuration, which is consistent with a real Marriott webpage rather than a spoof. I did not find any password fields, login forms, credential-posting actions, wallet-connect prompts, fake CAPTCHA, browser update lures, hidden iframes, or obfuscated credential-harvesting JavaS


### 5. GPT-5.4 — sample 3975

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: ``  |  **GT**: 0  |  **correct**: 1
- **Confidence**: 0.97  |  **Trust**: 0.83  |  **n_found/n_not/n_contr**: 14/15/5
- **Reasoning**: The HTML is consistent with a legitimate Legacy.com funeral-home profile page: the title, meta description, canonical URL, Open Graph tags, schema.org LocalBusiness/Event markup, and business contact details all describe Robinson & Son Mortuary Inc in Greenwood, SC. I found numerous standard analytics/ad scripts (Google Tag Manager, Bing, Microsoft Clarity, Criteo, ScorecardResearch) but no login form, no password field, no payment/crypto request, no credential-posting JavaScript, and no brand i


### 6. GPT-5.4 — sample 4048

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: ``  |  **GT**: 0  |  **correct**: 1
- **Confidence**: 0.98  |  **Trust**: 0.83  |  **n_found/n_not/n_contr**: 8/12/5
- **Reasoning**: The HTML is a Facebook page on the real Facebook infrastructure: the document root has id="facebook", canonical and Open Graph URLs point to https://www.facebook.com/p/noicompriamoautoit-100066769592620/, and assets are loaded from facebook-controlled fbcdn domains. The content describes a public profile/page for "noicompriamoauto.it" with social metadata and app links, not a login or payment page, and there are no password fields, credential submission forms, wallet prompts, fake update text, o


---

## cross_model_split  (243 total)

### 1. DeepSeek-V4-Flash — sample 1430

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: ``  |  **GT**: 1  |  **correct**: 0
- **Confidence**: 0.99  |  **Trust**: 0.87  |  **n_found/n_not/n_contr**: 0/13/1
- **Reasoning**: The URL domain is imdb.com, a top-250 Tranco site with a 30-year domain age and extensive Wayback history, strongly indicating legitimacy. The HTML content shows standard IMDb meta tags, Open Graph properties, and Amazon analytics scripts, with no password fields, login forms, credential-harvesting JavaScript, hidden iframes, or brand impersonation. The page title 'Private list' and description 'The creator of this list has not enabled public viewing' match a legitimate IMDb feature for private 


### 2. GPT-5.4 — sample 1430

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: ``  |  **GT**: 1  |  **correct**: 0
- **Confidence**: 0.98  |  **Trust**: 0.86  |  **n_found/n_not/n_contr**: 10/12/1
- **Reasoning**: The HTML is consistent with a legitimate IMDb page: it contains IMDb/OpenGraph metadata (`og:site_name=IMDb`, `twitter:site=@IMDb`, `imdb:pageType=list`) and Amazon-owned infrastructure references such as `images-na.ssl-images-amazon.com`, `m.media-amazon.com`, `fls-na.amazon.com`, and `unagi-na.amazon.com`. The visible page purpose is a normal access-controlled message, with title `Private list` and description `The creator of this list has not enabled public viewing`, not a login or payment pr


### 3. GPT-5.4-mini — sample 1430

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: ``  |  **GT**: 1  |  **correct**: 0
- **Confidence**: 0.99  |  **Trust**: 0.87  |  **n_found/n_not/n_contr**: 4/5/1
- **Reasoning**: The URL/context points to IMDb, a highly established domain with strong OSINT support for legitimacy: Tranco top-1k ranking, 30.4-year domain age, and extensive Wayback history. In the HTML, I found an IMDb page titled 'Private list' with standard IMDb/OpenGraph metadata and normal Amazon/IMDb scripts; I did not find password fields, login forms, credential-harvest JavaScript, hidden iframes, fake CAPTCHA, redirects, or brand-impersonation cues beyond the genuine IMDb branding. The page content 


### 4. GPT-5.1 — sample 1430

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: ``  |  **GT**: 1  |  **correct**: 0
- **Confidence**: 0.95  |  **Trust**: 0.85  |  **n_found/n_not/n_contr**: 5/10/1
- **Reasoning**: The HTML clearly belongs to IMDb (an Amazon property), with consistent scripts, telemetry domains, meta tags, and branding, and the Tranco/OSINT data shows a very old, highly popular domain, all of which strongly indicate a legitimate site. The page content is a standard 'Private list' informational page with no visible forms, no password or credential fields, and no signs of user-input harvesting or scam content. All loaded third‑party resources (Amazon/IMDB telemetry, scorecardresearch, privac


### 5. Llama-3.3-70B — sample 1430

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: ``  |  **GT**: 1  |  **correct**: 0
- **Confidence**: 0.95  |  **Trust**: 0.85  |  **n_found/n_not/n_contr**: 5/5/1
- **Reasoning**: The webpage has a long domain age of 30.4 years, a high Tranco rank of #244, and a rich archival history with 1000 snapshots on the Wayback Machine, indicating a legitimate and established website. The HTML content contains scripts from reputable sources such as Amazon and IMDb, and meta tags with proper OpenGraph protocol and Twitter card settings, further suggesting a genuine webpage. No suspicious indicators such as password fields, hidden iframes, or obfuscated content were found.


### 6. Mistral-Large-3 — sample 1430

- **URL**: `nan`
- **Domain**: `nan`
- **Label LLM**: ``  |  **GT**: 1  |  **correct**: 0
- **Confidence**: 0.99  |  **Trust**: 0.93  |  **n_found/n_not/n_contr**: 8/13/0
- **Reasoning**: The domain 'imdb.com' is a well-established, top-ranked site (Tranco #244) with a 30-year history, long-standing RDAP records, and consistent archival snapshots. The HTML content matches IMDb's legitimate structure, including verified meta tags (e.g., 'google-site-verification'), no credential-harvesting forms, and third-party scripts from known vendors (e.g., Amazon, ScorecardResearch). OSINT signals (domain age, registrar legitimacy, ASN longevity) further corroborate authenticity.


---
