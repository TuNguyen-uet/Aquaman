# class Document:
#
# class Sentence:
#
# class Token:


class Input:
    def __init__(self, text):
        self.text = text


class AspectOutput:
    def __init__(self, aspects, scores):
        self.aspects = aspects
        self.scores = scores


class PolarityOutput:
    def __init__(self, labels, aspects, scores):
        self.labels = labels
        self.aspects = aspects
        self.scores = scores
