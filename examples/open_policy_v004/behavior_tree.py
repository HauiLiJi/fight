from enum import Enum


class Status(Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class Condition:
    def __init__(self, predicate):
        self.predicate = predicate

    def tick(self, context):
        return Status.SUCCESS if self.predicate(context) else Status.FAILURE


class Action:
    def __init__(self, operation):
        self.operation = operation

    def tick(self, context):
        self.operation(context)
        return Status.SUCCESS


class Sequence:
    def __init__(self, *children):
        self.children = children

    def tick(self, context):
        for child in self.children:
            if child.tick(context) is Status.FAILURE:
                return Status.FAILURE
        return Status.SUCCESS


class Selector:
    def __init__(self, *children):
        self.children = children

    def tick(self, context):
        for child in self.children:
            if child.tick(context) is Status.SUCCESS:
                return Status.SUCCESS
        return Status.FAILURE
