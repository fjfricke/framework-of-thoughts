from llm_graph_optimizer.measurement.measurement import MeasurementsWithCache


class OperationFailed(Exception):
    from llm_graph_optimizer.measurement.measurement import Measurement
    """
    Exception raised when an operation fails.
    """
    def __init__(self, message: str, measurement: Measurement | MeasurementsWithCache = Measurement()):
        super().__init__(message)
        self.measurement = measurement

class GraphExecutionFailed(Exception):
    from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
    """
    Exception raised when a graph execution fails.
    """
    def __init__(self, message: str, process_measurement: ProcessMeasurement):
        super().__init__(message)
        self.process_measurement = process_measurement