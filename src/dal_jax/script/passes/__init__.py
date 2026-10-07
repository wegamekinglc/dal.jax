"""Analysis passes over parsed events, ports of DAL's ``script/visitor`` processors.

They run in DAL's order: :mod:`varindex` numbers variables, :mod:`ifmeta`
records what each ``IF`` writes, :mod:`constfold` marks constants,
:mod:`domain` proves conditions always true/false (and sets fuzzy bounds),
and :mod:`constcond` folds the proven conditions away.
"""
