#EmptyLink
#An empty pipeline chain link, it does nothing to the data and takes no initialisation params
#Can be used to inherit default behaviour for quick prototyping

from src.PipelineLink import PipelineLink

class EmptyLink(PipelineLink):
    def __init__(self):
        pass
        
    #default implementation 
    def next_audio(self, packet):
        """Passes to next link if available, does nothing"""
        if self.callback is None:
            return 
        else:
            self.callback.next_audio(packet)
    
    def configuration_parameters(self) -> dict[str, any]:
        """Returns empty dict, because there are no parameters here"""
        return {}

