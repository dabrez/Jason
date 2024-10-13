import abc
class GeminiCallBack():
    @abc.abstractmethod()
    def onSuccess(response):
        """What should happen if Quiz JSON is successfully generated"""
        raise NotImplementedError() # Must be implemented
    
    @abc.abstractmethod()
    def onFailure():
        """What should happen if Quiz JSON fails to generate"""
        raise NotImplementedError()