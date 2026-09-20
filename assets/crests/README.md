# Team crests

This app displays a real crest for any team with an image file here,
falling back to a generated shield badge otherwise. All 20 files for the
current Premier League roster are present and committed to this repo, so
you'll see real crests throughout the app as shipped. The generated
shield badge is still the fallback for any future roster change
(promotion/relegation) until a matching file is added here.

To replace or extend these yourself: drop an image file in this folder
using the naming convention below. The app picks it up automatically, no
code changes needed, one line of code checks for each team's file at
render time and falls back to the generated badge if none is found.

## Naming

Files must be named `<slug>.<ext>` where `<ext>` is `png`, `svg`, `jpg`,
`jpeg`, or `webp`, and `<slug>` is the team name lowercased with every
run of non-alphanumeric characters replaced by a single underscore
(`src.app._team_slug`). For the current 20-team Premier League roster:

| Team | Expected filename |
|---|---|
| Arsenal | `arsenal.png` |
| Aston Villa | `aston_villa.png` |
| Bournemouth | `bournemouth.png` |
| Brentford | `brentford.png` |
| Brighton | `brighton.png` |
| Chelsea | `chelsea.png` |
| Coventry | `coventry.png` |
| Crystal Palace | `crystal_palace.png` |
| Everton | `everton.png` |
| Fulham | `fulham.png` |
| Hull | `hull.png` |
| Ipswich | `ipswich.png` |
| Leeds | `leeds.png` |
| Liverpool | `liverpool.png` |
| Man City | `man_city.png` |
| Man United | `man_united.png` |
| Newcastle | `newcastle.png` |
| Nott'm Forest | `nott_m_forest.png` |
| Sunderland | `sunderland.png` |
| Tottenham | `tottenham.png` |

Any other file extension in the list above works too (`arsenal.svg`,
`arsenal.webp`, etc.), the app checks each extension in turn and uses
whichever one it finds first.

## On the images included here

Official club crests are trademarks the clubs and the Premier League
actively enforce. Displaying them to identify each team (this row is
about Arsenal) is normal, common practice across football apps and sites;
that's the use here, not any claim of affiliation with or endorsement by
the clubs or the league. If you fork this project, sourcing and
committing your own copies is your own call to make, same as it was here.
