---
name: source-extraction-sql
description: Canonical Presto/Trino SQL that extracts the three regional GitHub repo dumps from GitHub's hive warehouse. Verified against the actual notebooks in naics-repo-classification/1_scripts/.
metadata:
  type: project
---

The three regional source parquets in
`naics-github-classifier/data/raw/` were produced by running a
Presto/Trino SQL against GitHub's internal `hive` warehouse from
`C:\Users\Alexander\Dropbox\github_laptop\GitHub\naics-repo-classification\1_scripts\10_sql_extract_data.ipynb`.

**Data freshness:** April 15, 2025 (US presample); August 15, 2025
(EU readme extraction). Final regional parquets sit on top of those
snapshots.

**Canonical SQL (one run per region):**

```sql
SELECT
    nwo, num_stars, is_fork, is_archived, has_readme,
    content_size AS readme_content_size, disk_usage_bytes, spdx_license,
    repositories_current.id AS repository_id,
    from_utf8(repositories_current.description) AS description,
    repositories_current.topics,
    from_utf8(readme_current.content) AS readme_content,
    accounts_current.country_account AS country_code,
    COUNT(DISTINCT issues_current.id) AS num_issues,
    COUNT(DISTINCT issues_current.user_dotcom_id) AS num_issue_authors,
    COUNT(DISTINCT github_collab_commit_contributions.user_id) AS num_commit_contributors,
    SUM(github_collab_commit_contributions.commit_count) AS num_commits,
    MAX(github_collab_commit_contributions.created_at) AS last_commit_at
FROM hive.canonical.repositories_current
JOIN hive.canonical.accounts_current
    ON repositories_current.owner_dotcom_id = accounts_current.dotcom_id
LEFT JOIN hive.canonical.issues_current
    ON issues_current.repository_id = repositories_current.id
    AND NOT issues_current.is_user_hidden
JOIN hive.suez.readme_current
    ON readme_current.repository_id = repositories_current.id
    AND LOWER(path) = 'readme.md'
    AND LOWER(filename) = 'readme.md'
LEFT JOIN hive.reference.licenses
    ON licenses.id = repositories_current.license_id
LEFT JOIN delta.snapshots.github_collab_commit_contributions
    ON github_collab_commit_contributions.repository_id = repositories_current.id
WHERE NOT is_spammy_owner
    AND is_public = TRUE
    AND has_readme = TRUE
    AND num_stars >= 1
    AND is_fork = FALSE
    AND accounts_current.country_account IN ( /* per-region list */ )
    AND content_size >= 750
    AND disk_usage_bytes > 0
    AND repositories_current.description IS NOT NULL
    AND TRIM(from_utf8(repositories_current.description)) != ''
GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12
HAVING SUM(github_collab_commit_contributions.commit_count) > 5
    AND COUNT(DISTINCT github_collab_commit_contributions.user_id) >= 2
ORDER BY repository_id
```

**Country lists:**
- USA: `'US'`
- EU (27 codes, no UK): `'AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK'`
- AU: `'AU'`

**Jurisdiction logic:** `accounts_current.country_account` on the
**owner account**, joined via `owner_dotcom_id = dotcom_id`. Not
top-contributor location or geo-IP — explicitly the owner-account
country.

**❗ Why this differs from Kevin's earlier verbal SQL:** The version
Kevin pasted as plain text in an earlier session
(`num_stars>=5`, `content_size>=1024`, no country/contributors
clauses) was a simplification. The actual production SQL above has
`num_stars>=1`, `content_size>=750`, an explicit country list, a
`HAVING >= 2 contributors`, a `HAVING > 5 commits`, a non-empty
description requirement, and a `disk_usage_bytes > 0` filter. The
canonical SQL is what produced the released regional parquets.

**Why this matters:** The paper's Source Pool description (P0-7 in
`paper/IMPROVEMENTS.md`) must reflect the canonical SQL. The
simplified version would be misleading to reviewers because it
implies a different sample frame.

**How to apply:** Use the canonical SQL in the paper's Section 3 and
in the Datasheet's "Collection Process" section. Acknowledge the
filter set explicitly (`>=2 contributors`, `>=6 commits`, `>=750`
byte README, non-empty description) so reviewers can judge selection
bias.

**Variants and ambiguities to verify with Kevin/Alexander before final
submission:**
1. Whether the USA SQL run also included the `HAVING >= 2 contributors`
   clause (the AU cell does; the US-only counting cell I read did not).
2. Whether the `other_lang_md == 1` filter (English-language README
   flag) seen in the pandas grid-search notebook was applied to the
   final regional parquets.
3. The `>= 2 contributors` could have been applied either in the
   HAVING clause (AU) or as a downstream pandas filter (US). Either
   way the filter is active in the released v1.0 — confirmed.
