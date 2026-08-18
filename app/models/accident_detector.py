class AccidentDetector:
    """
    Placeholder cho mô hình phát hiện tai nạn (Accident).
    Theo yêu cầu, module này DISABLED.
    """
    def __init__(self, model_path=None):
        self.enabled = False
        
    def predict(self, *args, **kwargs):
        if not self.enabled:
            return None, 0.0
        pass
