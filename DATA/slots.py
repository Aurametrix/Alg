class PythonClass():
    """This is a simple Python class"""
    
    def __init__(self, message):
        """Init method, nothing special here"""
        self.message = message
        self.capital_message = self.make_it_bigger()
    
    def make_it_bigger(self):
        """Return a capitalized version of message"""
        return self.message.upper()
    
    def scream_message(self):
        """Print the capital_message attribute of the instance"""
        print(self.capital_message) 
        
my_instance = PythonClass("my message")
