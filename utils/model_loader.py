import os
from dotenv import load_dotenv
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field
from utils.config_loader import load_config
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

class ConfigLoader:
    def __init__(self):
        '''
        For loading configurations
        '''
        print("Loading config.........")
        self.config = load_config()

    def __getitem__(self, key):
        '''
        __getitem__ is a dunder method in Python.
        It lets your class behave like a dictionary or list, so you can use square brackets [] to access items.
        '''
        return self.config[key]
    

class ModelLoader(BaseModel):
    '''
    You are defining a class called ModelLoader.It inherits from BaseModel, which comes from Pydantic.
    BaseModel allows you to define data models with validation, type checking, and serialization.
    '''
    model_provider: Literal["groq","openai"] = "groq"
    '''
    Its type is Literal["groq","openai"]:This means it can only take one of these two string values: "groq" or "openai".
    If you try to assign anything else (like "anthropic"), Pydantic will raise a validation error.
    The default value is "groq".
    So if you create ModelLoader() without specifying model_provider, it will automatically be "groq".
    '''
    config:Optional[ConfigLoader] = Field(default = None, exclude=True)
    '''
    config is another field of the class.
    Its type is Optional[ConfigLoader]:
    Optional means it can either be a ConfigLoader object or None.
    The default value is None.
    Field(...) is a Pydantic helper to add extra metadata:
    default=None → initializes it as None if not provided.
    exclude=True → tells Pydantic not to include this field when exporting the model (e.g., .dict() or .json()).
    '''
    def model_post_init(self, __context:Any)->None:
        self.config = ConfigLoader()
        '''
        What is model_post_init?
        In Pydantic v2, model_post_init is a lifecycle hook.
        It’s called automatically after Pydantic finishes validating and initializing your model.
        It’s the right place to run custom initialization logic (like setting defaults, loading configs, or transforming fields).
        Think of it like Python’s __post_init__ in dataclasses, but for Pydantic models.
        self → the instance of your model (ModelLoader in your case).
       
        __context is an internal argument that Pydantic passes during model creation.
        It’s mainly used inside Pydantic itself to carry extra information between validation/initialization steps.
        Most of the time, you don’t need it in your own code.
        '''

    class Config:
        arbitrary_types_allowed = True
        '''
        arbitrary_types_allowed = True is a setting that tells Pydantic:
        “Allow me to use non-Pydantic, arbitrary Python classes as fields in my model.”
        '''
    
    def load_llm(self):
        '''
        Load and return the LLM model
        '''
        print("LLM Loading......")
        print(f"Loading the model from the provider : {self.model_provider}")
        if self.model_provider=="groq":
            print("Loading LLM from groq......")
            groq_api_key = os.getenv("GROQ_API_KEY")
            model_name = self.config["llm"]["groq"]["model_name"]
            llm = ChatGroq(model=model_name, api_key = groq_api_key)
        elif self.model_provider=="openai":
            print("Loading LLM from OpenAI.......")
            openai_api_key=os.getenv["OPENAI_API_KEY"]
            model_name = self.config["llm"]["openai"]["model_name"]
            llm = ChatOpenAI(model_name="o4-mini", api_key = openai_api_key)
        return llm




    
    


    

