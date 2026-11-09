# Releasing

**Every release ships its change-list.**
- [CHANGELOG.md](CHANGELOG.md) holds one entry per release.
- The GitHub release notes and any announcement of the release are written from that entry.
- An announcement says nothing the entry does not.

## The steps, in order

1. **CHANGELOG.md first.**
   - Move the `[Unreleased]` items under a new heading, `## [X.Y.Z] - YYYY-MM-DD`.
   - Use the Keep a Changelog sections: Added, Changed, Deprecated, Removed, Fixed, Security.
   - Describe what a user sees.
   - Update the compare links at the bottom.
2. **CITATION.cff:** set `version` to `X.Y.Z` and `date-released` to the release date.
3. **The gate:** `bash ci/gates.sh` passes.
4. **Tag** the release commit `vX.Y.Z`, and create the GitHub release from that tag.
5. **The notes:** the GitHub release's notes are the CHANGELOG entry for `X.Y.Z`.

A release's version is its git tag, `vX.Y.Z`.
