"""Promo filter: detect scanlator promo pages/slices against user-supplied examples via perceptual hash.

Import the submodules directly (`omniscan.filter.decide`, `.apply`, `.hashing`): this package re-exports
nothing, so importing `decide` (the web server does) never pulls torch in through `apply`.
"""
