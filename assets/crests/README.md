# Team crests (optional, not included)

This project deliberately does not ship or fetch any club crest artwork
(see the main README for why). If you want real crests instead of the
generated shield badges, drop image files in this folder yourself. The
app picks them up automatically, no code changes needed, one line
of code checks for each file at render time and falls back to the
generated badge for any team without one.

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

## Sourcing images

That part is on you. Whatever you put here gets embedded directly into
the rendered app (and, if you commit this folder, into your public repo),
so you're taking on whatever licensing/reproduction risk that carries.
Official club crests are trademarks the clubs and the Premier League
actively enforce; this project isn't going to tell you that's fine, only
that the technical wiring is ready if you decide it's a risk you're
willing to take for your own copy of the project.
