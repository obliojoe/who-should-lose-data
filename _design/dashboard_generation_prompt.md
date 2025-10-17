# NFL Dashboard Content Generation Prompt

## Objective
Generate a comprehensive `dashboard_content.json` file that powers the NFL playoff dashboard homepage. This JSON provides AI-generated insights, statistical analysis, and compelling narratives about the current NFL season.

## How This Prompt Works

This prompt will be used with **inline data** provided in JSON format. The data will be pre-selected and formatted to include only the most relevant information needed for dashboard generation, keeping the prompt within context limits.

The prompt will follow this structure:
```
[This instruction document]
---
DATA FOR WEEK {N}:
[Pre-formatted JSON with curated data]
---
Generate the dashboard_content.json based on the above data.
```

## Data Format You'll Receive

You will receive a JSON object with the following structure:

```json
{"current_week":5,"timestamp":"2025-10-08T13:19:23.923016","teams":{"DET":{"record":"4-1","playoff_probability":96.2,"division_probability":78.5,"current_seed":2,"sagarin_rating":25.64,"sagarin_rank":3,"stats":{"points_per_game":34.8,"points_allowed_per_game":22.4,"total_yards_per_game":372.8,"rushing_yards_per_game":134.8,"passing_yards_per_game":238.0,"completion_pct":75.3,"third_down_pct":40.0,"red_zone_pct":72.0,"turnover_margin":2.0},"top_players":[{"name":"Jared Goff","position":"QB","stats":"1,190 pass yds, 13 TDs, 2 INTs"}]}},"upcoming_games":[{"espn_id":"401772918","week":6,"away_team":"BAL","home_team":"BUF","game_date":"2025-10-13","is_divisional":false,"is_thursday":false}],"recent_results":[{"espn_id":"401772510","week":5,"away_team":"DAL","away_score":20,"home_team":"PHI","home_score":24}],"power_rankings_previous_week":{"PHI":1,"BUF":2,"DET":4}}
```

## Output JSON Schema

```json
{
  "timestamp": "<ISO 8601 timestamp>",
  "week": <current NFL week number>,
  "league_pulse": {
    "summary": "<2-3 sentence markdown summary>",
    "key_storylines": [
      {
        "title": "<Catchy headline>",
        "description": "<1-2 sentence description with stats>"
      }
      // Exactly 3 storylines
    ]
  },
  "game_of_the_week": {
    "espn_id": "<ESPN game ID from schedule.json>",
    "away_team": "<3-letter team abbreviation>",
    "away_record": "<W-L or W-L-T>",
    "home_team": "<3-letter team abbreviation>",
    "home_record": "<W-L or W-L-T>",
    "tagline": "<1-2 sentence compelling description of why this is the top game>",
    "betting_line": "<Optional: e.g., 'KC -2.5'>",
    "over_under": <Optional: total points line>
  },
  "game_of_the_meek": {
    "espn_id": "<ESPN game ID from schedule.json>",
    "away_team": "<3-letter team abbreviation>",
    "away_record": "<W-L or W-L-T>",
    "home_team": "<3-letter team abbreviation>",
    "home_record": "<W-L or W-L-T>",
    "tagline": "<Humorous 1-2 sentence explanation of why this game is skippable>",
    "betting_line": "<Optional: e.g., 'CAR -3'>",
    "over_under": <Optional: total points line>
  },
  "stat_leaders": {
    "offense": [
      {
        "team": "<3-letter abbreviation>",
        "stat": "<Stat name>",
        "value": <numerical value>,
        "unit": "<Optional: '%', ' yds', ' sacks', etc.>",
        "rank": <league rank 1-32>,
        "context": "<Short phrase explaining significance>"
      }
      // Top 5 offensive stats
    ],
    "defense": [/* Same structure, 5 entries */],
    "efficiency": [/* Same structure, 5 entries */]
  },
  "individual_highlights": [
    {
      "player": "<Full name>",
      "team": "<3-letter abbreviation>",
      "position": "<Position abbreviation>",
      "stat_line": "<Brief stat summary>",
      "context": "<Why this performance matters>",
      "season_projection": "<Optional: What pace for full season>"
    }
    // 6 individual highlights
  ],
  "power_rankings": {
    "top_5": [
      {
        "rank": <1-5>,
        "team": "<3-letter abbreviation>",
        "rating": <Sagarin rating>,
        "record": "<W-L-T>",
        "trend": "<'up' | 'down' | 'steady'>"
      }
    ],
    "bottom_5": [/* ranks 28-32 */],
    "biggest_riser": {
      "team": "<3-letter abbreviation>",
      "previous_rank": <number>,
      "current_rank": <number>,
      "movement": "<e.g., '+3'>",
      "rating": <Sagarin rating>,
      "reason": "<1 sentence explanation>"
    },
    "biggest_faller": {/* Same structure as biggest_riser */}
  },
  "playoff_snapshot": {
    "top_4": [
      {
        "team": "<3-letter abbreviation>",
        "probability": <playoff probability>,
        "current_seed": <number or null>,
        "record": "<W-L or W-L-T>",
        "key_info": "<Clinch scenario or key context>"
      }
      // Top 4 teams by playoff probability
    ],
    "middle_4": [
      {
        "team": "<3-letter abbreviation>",
        "probability": <playoff probability>,
        "current_seed": <number or null>,
        "record": "<W-L or W-L-T>",
        "key_info": "<Key upcoming games or context>"
      }
      // Next 4 teams by playoff probability (bubble teams)
    ],
    "bottom_4": [
      {
        "team": "<3-letter abbreviation>",
        "probability": <playoff probability>,
        "current_seed": null,
        "record": "<W-L or W-L-T>",
        "key_info": "<Elimination scenario or context>"
      }
      // Bottom 4 teams by playoff probability (longshots)
    ]
  },
  "week_preview": {
    "thursday_night": {
      "espn_id": "<ESPN game ID>",
      "away_team": "<3-letter abbreviation>",
      "home_team": "<3-letter abbreviation>",
      "tagline": "<Brief compelling description>"
    },
    "sunday_spotlight": [
      {
        "espn_id": "<ESPN game ID>",
        "away_team": "<3-letter abbreviation>",
        "home_team": "<3-letter abbreviation>",
        "tagline": "<Brief compelling description>"
      }
      // 2-3 most compelling Sunday games
    ],
    "total_games": <number of games this week>
  }
}
```

## Detailed Instructions by Section

### 1. League Pulse

**Purpose**: Provide a high-level overview of the league's current state.

**Summary Requirements**:
- 2-3 sentences in markdown format
- Use **bold** for emphasis on key points (teams, records, trends)
- Include at least 2-3 specific statistics
- Capture the most compelling storylines (playoff races, surprises, collapses)
- Be analytical but engaging

**Format**:
```markdown
"<Engaging 2-3 sentence summary highlighting major conference/division dynamics, surprise teams, and playoff implications. Use **bold** for team names and include 2-3 specific statistics.>"
```

**Key Storylines Requirements**:
- Exactly 3 storylines
- Each with a catchy, concise title (3-6 words)
- Description: 1-2 sentences with specific stats/records
- Cover different types of stories: division races, surprise teams, player performances, playoff implications, statistical anomalies

**Format**:
```json
{
  "title": "<Catchy 3-6 word headline>",
  "description": "<1-2 sentences with team records and specific statistics explaining the storyline>"
}
```

### 2. Game of the Week

**Purpose**: Identify the single most compelling game this week and include its full preview analysis.

**Selection Criteria** (in order of importance):
1. **Playoff implications** - Games between teams with >30% playoff probability
2. **Power ranking matchup** - Top 10 teams facing each other
3. **Division rivalry** - Same division games
4. **Tiebreaker implications** - Conference/head-to-head matchups
5. **Narratives** - Revenge games, streaks, coaching storylines

**Tagline Requirements**:
- 1-2 sentences that capture why this is THE game to watch
- Should highlight the stakes, storylines, or star power
- Compelling and engaging, not just descriptive

**Optional Fields**:
- `betting_line`: Include if available (e.g., "KC -2.5")
- `over_under`: Include if available (e.g., 52.5)

### 2b. Game of the Meek

**Purpose**: Identify the least compelling game this week with humor.

**Selection Criteria** (what makes it "meek"):
1. **Both teams have low playoff probability** (<10%)
2. **Both teams ranked bottom-10 in power rankings**
3. **Lopsided matchup** - One team heavily favored
4. **Low stakes** - No divisional or playoff implications
5. **Unexciting teams** - Poor records, boring playstyles, low-scoring

**Tagline Requirements**:
- 1-2 sentences, humorous but not mean-spirited
- Explain WHY it's skippable (boring matchup, low stakes, blowout expected)
- Be creative and entertaining

**Optional Fields**:
- `betting_line`: Include if available
- `over_under`: Include if available

### 3. Stat Leaders

**Purpose**: Showcase top performing teams in three categories.

**Categories**:
- **Offense**: Points/game, yards/game, passing yards, rushing yards, red zone TD%, completion%, etc.
- **Defense**: Points allowed/game, sacks/game, takeaways, third down defense%, red zone defense%, etc.
- **Efficiency**: Third down%, turnover margin, EPA/game, red zone efficiency%, fourth down%, etc.

**Requirements**:
- Exactly 5 stats per category
- Diverse stats (don't repeat similar stats)
- Use actual rank-1 leaders from team_stats.json when possible
- **Include `unit` field** for proper display ("%", " yds", " sacks", etc.)
- Context should be concise (2-6 words) and add color/explanation
- Value should be precise (use decimal precision from data)

**Unit Examples**:
- Percentage stats: `"unit": "%"` → displays as "34.8%"
- Yards: `"unit": " yds"` → displays as "1190.0 yds"
- Counts (sacks, takeaways): `"unit": ""` or omit → displays as "11.0"
- Per game stats: usually no unit needed if already in stat name

**Context Guidelines**:
- 2-6 words that add color or explanation
- Can reference star players, team identity, or statistical significance
- Avoid generic phrases - be specific to the team/stat

**Format**:
```json
{
  "team": "<3-letter abbr>",
  "stat": "<Stat name>",
  "value": <number>,
  "unit": "<%, yds, sacks, or empty string>",
  "rank": <1-32>,
  "context": "<Brief 2-6 word context>"
}
```

### 4. Individual Highlights

**Purpose**: Celebrate standout individual performances.

**Selection Criteria**:
- Statistical leaders (rushing yards, TDs, passing yards, sacks, interceptions)
- Players on winning/playoff teams preferred but not required
- Diverse positions (at least 3 different positions)
- Recent hot streaks or season-long excellence
- Mix of established stars and breakout performers

**Requirements**:
- Exactly 6 players
- Stat line should be concise but complete
- Context explains WHY the performance matters (team success, pace, league-leading, etc.)
- Season projection is optional but adds value (calculate pace for full 17-game season)

**Format**:
```json
{
  "player": "<Full player name>",
  "team": "<3-letter abbr>",
  "position": "<QB/RB/WR/TE/etc>",
  "stat_line": "<Concise stat summary>",
  "context": "<Why this performance matters>",
  "season_projection": "<Optional: Projected full-season stats>"
}
```

### 5. Power Rankings

**Purpose**: Show team quality rankings based on Sagarin ratings.

**Data Source**: Use `sagarin.json` ratings snapshot

**Requirements**:

**Top 5**:
- Ranks 1-5 from Sagarin ratings
- Include exact rating value
- Record from standings_cache.json
- Trend: Compare to previous week if available, otherwise use recent performance
  - "up" = Won recently or improving
  - "down" = Lost recently or declining
  - "steady" = Consistent performance

**Bottom 5**:
- Ranks 28-32 from Sagarin ratings
- Same structure as top 5

**Biggest Riser/Faller**:
- Calculate movement from previous week's rankings (if available)
- If no previous week, identify teams with winning/losing streaks
- Reason should be 1 sentence explaining the change (specific wins/losses, injury impacts, statistical trends)

**Format**:
```json
{
  "biggest_riser": {
    "team": "<3-letter abbr>",
    "previous_rank": <number>,
    "current_rank": <number>,
    "movement": "<+N or -N>",
    "rating": <Sagarin rating>,
    "reason": "<One sentence explaining the movement with specific context>"
  }
}
```

### 6. Playoff Snapshot

**Purpose**: Show the playoff race by dividing teams into three tiers of 4 teams each.

**Data Source**: `analysis_cache.json` → team_analyses → playoff_chance

**Structure**: Fixed 4 teams per category, ordered by playoff probability

**Top 4 (Highest Playoff Probabilities)**:
- The 4 teams with the highest playoff probabilities
- These are likely playoff locks or near-locks
- `current_seed`: Current playoff seed (1-7) or null if not yet seeded
- `record`: Actual W-L or W-L-T from standings
- `key_info`: Brief context like "Cruising to division title" or "Commanding NFC East lead"

**Middle 4 (Bubble Teams)**:
- The next 4 teams by playoff probability (typically 40-70% range)
- Teams where every game matters - the true bubble
- `current_seed`: Current seed or null
- `record`: Actual W-L or W-L-T
- `key_info`: Mention 1-2 key upcoming opponents or context ("@ BAL, vs CIN crucial")

**Bottom 4 (Longshots)**:
- The 4 teams with lowest playoff probabilities that still have >0% chance
- Teams on life support, nearly eliminated
- `current_seed`: Should be null (they're out of playoffs)
- `record`: Actual W-L or W-L-T
- `key_info`: Brief elimination scenario ("Must win out", "Season slipping away")

**Important**: Each category must have exactly 4 teams. Select the 12 most interesting/relevant teams based on playoff probability rankings.

**Format**:
```json
{
  "top_4": [{
    "team": "<3-letter abbr>",
    "probability": <playoff probability>,
    "current_seed": <1-7 or null>,
    "record": "<W-L or W-L-T>",
    "key_info": "<Brief context: clinch scenario, key to success, etc>"
  }],
  "middle_4": [{
    "team": "<3-letter abbr>",
    "probability": <playoff probability>,
    "current_seed": <1-7 or null>,
    "record": "<W-L or W-L-T>",
    "key_info": "<Upcoming key games or bubble context>"
  }],
  "bottom_4": [{
    "team": "<3-letter abbr>",
    "probability": <playoff probability>,
    "current_seed": null,
    "record": "<W-L or W-L-T>",
    "key_info": "<Elimination scenario or 'must-win' context>"
  }]
}
```

### 7. Week Preview (Optional)

**Purpose**: Highlight key upcoming games.

**Requirements**:
- Include Thursday Night Football if there is one
- Select 2-3 Sunday spotlight games based on:
  - Playoff implications
  - Prime time games
  - Division matchups
  - Top-10 power ranking matchups
- Taglines should be brief (5-8 words) and compelling
- Total games = number of games in the upcoming week

**Format**:
```json
{
  "thursday_night": {
    "espn_id": "<ESPN game ID>",
    "away_team": "<3-letter abbr>",
    "home_team": "<3-letter abbr>",
    "tagline": "<Brief 5-8 word compelling description>"
  },
  "sunday_spotlight": [{
    "espn_id": "<ESPN game ID>",
    "away_team": "<3-letter abbr>",
    "home_team": "<3-letter abbr>",
    "tagline": "<Brief 5-8 word compelling description>"
  }]
}
```

## Tone and Style Guidelines

### Writing Style
- **Analytical but engaging** - Use statistics but tell stories
- **Bold for emphasis** - Use markdown **bold** to highlight key teams, players, records, trends
- **Concise** - Every word should add value
- **Specific** - Use exact numbers, not vague terms ("34.8 PPG" not "high-scoring")
- **Active voice** - "Detroit dominates" not "Detroit is dominating"

### Statistical Precision
- Use decimal precision from source data (34.8, not 35)
- Always include units (PPG, yards/game, %)
- Rank stats when relevant (#1 in NFL, league-leading)

### Context and Storytelling
- Connect stats to narratives (MVP campaigns, playoff races, streaks, breakouts)
- Use comparative language (elite, struggling, resurgent, collapsed)
- Reference specific players when it adds color ("Josh Allen's MVP campaign", "Saquon Barkley revolution")

### Team References
- Use 3-letter abbreviations in JSON fields (DET, BUF, KC)
- Use city names or nicknames in narrative text for variety
- Be consistent within each section

## Data Processing Notes

### Pre-Processed Data
All statistical calculations (points per game, yards per game, percentages) will be **pre-calculated** in the input JSON. You don't need to perform calculations - just select the most interesting stats and add context.

### Your Analytical Tasks
1. **Select the most compelling stats** from the provided data
2. **Add context** that explains why stats matter (player names, team situations, trends)
3. **Identify narratives** by comparing teams, finding outliers, noting streaks
4. **Determine trends** by comparing current vs previous week power rankings, recent game results
5. **Prioritize games** based on playoff probabilities, power rankings, divisional matchups

### Missing Data Handling
- If betting lines aren't available, omit those fields
- If previous week rankings aren't available, use win/loss streaks for trend
- Week preview is optional - include if meaningful games exist
- Season projections are optional for individual highlights

### What Makes a Good Selection

**For Stat Leaders:**
- Pick stats that tell a story (extreme highs/lows, surprising leaders)
- Diverse stats across categories
- Mix of well-known teams and surprises

**For Game of the Week:**
- Prioritize games with combined playoff probability >100%
- Top-15 power ranking matchups
- Division games between competitive teams
- Recent streaks or revenge narratives

**For Individual Highlights:**
- Statistical outliers (top 3 in major categories)
- Players on winning teams preferred
- Mix positions (QB, RB, WR, defensive stars)
- Breakout performances, not just established stars

## Data Preparation Guidelines (For Backend Implementation)

When preparing the inline data JSON to send with this prompt, include:

### Required Data Per Team (All 32 Teams)
- Current record (W-L-T format)
- Playoff probability, division probability, current playoff seed
- Sagarin rating and rank
- Key stats (10-15 most interesting stats):
  - Offensive: PPG, yards/game, passing yards, rushing yards, completion %, red zone %
  - Defensive: Points allowed/game, sacks/game, takeaways, third down %, red zone %
  - Efficiency: Third down %, turnover margin, EPA
- Top 2-3 players with season stat lines (focus on QBs, RBs, WRs, defensive stars)

### Games Data
- **Upcoming week games**: ESPN ID, teams, date, flags (is_divisional, is_thursday, is_primetime)
- **Recent results** (last 1-2 weeks): ESPN ID, teams, scores for trend analysis

### Power Rankings History
- Previous week's Sagarin rankings (at minimum, just the rank order) to calculate movers

### Optional but Valuable
- Win/loss streaks
- Injury notes for star players
- Recent coaching changes or major storylines
- Betting lines if available

### Data Size Strategy
- **Focus on quality over quantity**: Better to have 15 meaningful stats per team than 50 fields
- **Pre-filter players**: Only include statistical leaders or players on playoff-contending teams
- **Limit games**: Only upcoming week + previous week, not full season schedule
- **Calculate derived stats**: PPG, yards/game, etc. before sending to save tokens

## Example Output Quality Check

Before finalizing, ensure:
- ✅ All required fields are present with correct data types
- ✅ Markdown formatting is used in summary, reason, and description fields
- ✅ Statistics are specific and accurate from source data
- ✅ Team abbreviations are valid 3-letter codes
- ✅ Probabilities are between 0-100
- ✅ ESPN IDs match actual games in schedule.json
- ✅ Records format is "W-L" or "W-L-T"
- ✅ Context phrases are concise and add value
- ✅ Narratives are compelling and data-driven
- ✅ No duplicate teams within same category (unless warranted)
- ✅ Timestamp is current ISO 8601 format
- ✅ Week number matches current NFL week
