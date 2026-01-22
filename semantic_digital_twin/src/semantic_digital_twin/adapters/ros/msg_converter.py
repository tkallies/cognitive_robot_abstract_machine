from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Type

from typing_extensions import Generic, TypeVar, ClassVar, Any

from krrood.utils import recursive_subclasses, DataclassException
from semantic_digital_twin.world import World

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


@dataclass
class ROS2ConversionError(DataclassException):
    message: str = field(init=False)


@dataclass
class CannotConvertSemDTToRos2Error(ROS2ConversionError):
    data_type: Type = field(kw_only=True)

    def __post_init__(self):
        self.message = f"Cannot convert {self.data_type.__name__} to ROS2 message."


@dataclass
class CannotConvertRos2ToSemDTError(ROS2ConversionError):
    data_type: Type = field(kw_only=True)

    def __post_init__(self):
        self.message = f"Cannot convert {self.data_type.__name__} to our semDT type."


@dataclass
class Ros2ToSemDTConverter(ABC, Generic[InputType, OutputType]):
    input_type: InputType
    """The semDT type for which this converter handles conversion."""
    output_type: OutputType
    """The ROS2 message type for which this converter handles conversion."""

    @classmethod
    def can_convert(cls, data: Any) -> bool:
        return cls.input_type == type(data)

    @classmethod
    def get_to_converter(cls, input_obj: Any) -> Type[Ros2ToSemDTConverter]:
        """
        Recursively checks all subclasses of Ros2ToSemDTConverter to find the converter for the given semDT type.
        :param our_type: The semDT type for which to find the ROS2 converter.
        :return: The Ros2ToSemDTConverter subclass that handles conversion from the given semDT type.
        """
        for sub_class in recursive_subclasses(cls):
            if sub_class.can_convert(input_obj):
                return sub_class
        raise CannotConvertSemDTToRos2Error(data_type=type(input_obj))

    @classmethod
    def convert(cls, data: InputType, world: World) -> OutputType:
        return cls.get_to_converter(data).convert(data, world)


@dataclass
class SemDTToRos2Converter(ABC, Generic[InputType, OutputType]):
    registry: ClassVar[Dict[Type, Type[SemDTToRos2Converter]]] = field(default={})
    """Registry mapping semDT types to their corresponding ROS2MessageConverter subclass."""

    input_type: InputType
    """The semDT type for which this converter handles conversion."""
    output_type: OutputType
    """The ROS2 message type for which this converter handles conversion."""

    @classmethod
    def can_convert(cls, obj: Any) -> bool:
        return cls.input_type == type(obj)

    @classmethod
    def get_to_converter(cls, input_obj: Any) -> Type[SemDTToRos2Converter]:
        """
        Recursively checks all subclasses of SemDTToRos2Converter to find the converter for the given semDT type.
        :param input_type: The semDT type for which to find the ROS2 converter.
        :return: The SemDTToRos2Converter subclass that handles conversion from the given semDT type.
        """
        for sub_class in recursive_subclasses(cls):
            if sub_class.can_convert(input_obj):
                return sub_class
        raise CannotConvertRos2ToSemDTError(data_type=type(input_obj))

    @classmethod
    def convert(cls, data: InputType) -> OutputType:
        return cls.get_to_converter(data).convert(data)
