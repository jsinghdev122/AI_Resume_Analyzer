"""Errors that mean "the input was unusable" (reported to the client as HTTP 400)."""


class InputError(ValueError):
    pass
