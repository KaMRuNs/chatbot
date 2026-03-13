class CustomException(Exception):
    """Base class for all custom exceptions."""
    pass

class NotFoundException(CustomException):
    """Exception raised for not found errors."""
    def __init__(self, message="Resource not found"):  
        self.message = message
        super().__init__(self.message)

class InvalidInputException(CustomException):
    """Exception raised for invalid input errors."""
    def __init__(self, message="Invalid input"):  
        self.message = message
        super().__init__(self.message)

class PermissionDeniedException(CustomException):
    """Exception raised for permission denied errors."""
    def __init__(self, message="Permission denied"):  
        self.message = message
        super().__init__(self.message)