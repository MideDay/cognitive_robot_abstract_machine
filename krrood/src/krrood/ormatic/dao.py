from __future__ import annotations

import abc
import inspect
import logging
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import _GenericAlias

import sqlalchemy.inspection
import sqlalchemy.orm
from sqlalchemy import Column
from sqlalchemy.orm import MANYTOONE, MANYTOMANY, ONETOMANY, RelationshipProperty
from typing_extensions import (
    Type,
    get_args,
    Dict,
    Any,
    TypeVar,
    Generic,
    Self,
    Optional,
    List,
    Iterable,
    Tuple,
)

from collections import deque
from .exceptions import (
    NoGenericError,
    NoDAOFoundError,
    NoDAOFoundDuringParsingError,
    UnsupportedRelationshipError,
)
from ..utils import recursive_subclasses

logger = logging.getLogger(__name__)
_repr_thread_local = threading.local()

T = TypeVar("T")
_DAO = TypeVar("_DAO", bound="DataAccessObject")
InstanceDict = Dict[int, Any]  # Dictionary that maps object ids to objects
InProgressDict = Dict[int, bool]


def is_data_column(column: Column):
    return (
        not column.primary_key
        and len(column.foreign_keys) == 0
        and column.name != "polymorphic_type"
    )


_WorkItemT = TypeVar("_WorkItemT", bound="DataAccessObjectWorkItem")


@dataclass
class DataAccessObjectWorkItem(abc.ABC):
    """
    Abstract base class for conversion work items.
    """

    dao_instance: DataAccessObject


@dataclass
class DataAccessObjectState(Generic[_WorkItemT], abc.ABC):
    """
    Abstract base class for conversion states.
    """

    memo: InstanceDict = field(default_factory=dict)
    """
    Dictionary that keeps track of already converted objects during DAO conversion.
    Maps object IDs to their corresponding DAO instances to prevent duplicate conversion
    and handle circular references. Acts as a memoization cache to improve performance
    and maintain object identity.
    """

    queue: deque[_WorkItemT] = field(default_factory=deque)
    """
    Queue of work items to be processed.
    """

    @abc.abstractmethod
    def add_to_queue(self, *args: Any, **kwargs: Any) -> None:
        """
        Add a new work item to the processing queue.
        """
        pass

    def has(self, source: Any) -> bool:
        """
        Check if the given source object has already been converted.
        """
        return id(source) in self.memo

    def get(self, source: Any) -> Optional[Any]:
        """
        Get the converted object for the given source object.
        """
        return self.memo.get(id(source))

    def register(self, source: Any, target: Any) -> None:
        """
        Register a conversion result in the memoization store.
        """
        self.memo[id(source)] = target

    def pop(self, source: Any) -> Optional[Any]:
        """
        Remove and return the conversion result for the given source object.
        """
        return self.memo.pop(id(source), None)


@dataclass
class ToDataAccessObjectWorkItem(DataAccessObjectWorkItem):
    """
    Work item for converting an object to a Data Access Object.
    """

    source_object: Any
    alternative_base: Optional[Type[DataAccessObject]] = None


@dataclass
class ToDataAccessObjectState(DataAccessObjectState[ToDataAccessObjectWorkItem]):
    """
    Encapsulates the conversion state for to_dao conversions.

    This bundles memoization and keep-alive dictionaries and exposes
    convenience operations used during conversion so that only the state
    needs to be passed around.
    """

    keep_alive: InstanceDict = field(default_factory=dict)
    """
    Dictionary that prevents objects from being garbage collected.
    """

    def add_to_queue(
        self,
        source_object: Any,
        dao_instance: DataAccessObject,
        alternative_base: Optional[Type[DataAccessObject]] = None,
    ):
        """
        Add a new work item to the processing queue.
        """
        self.queue.append(
            ToDataAccessObjectWorkItem(
                dao_instance=dao_instance,
                source_object=source_object,
                alternative_base=alternative_base,
            )
        )

    def apply_alternative_mapping_if_needed(
        self, dao_clazz: Type[DataAccessObject], source_object: Any
    ) -> Any:
        """
        Apply AlternativeMapping.to_dao if the dao class uses an alternative mapping.
        """
        if issubclass(dao_clazz.original_class(), AlternativeMapping):
            return dao_clazz.original_class().to_dao(source_object, state=self)
        return source_object

    def register(self, source_object: Any, dao_instance: DataAccessObject) -> None:
        """
        Register a partially built DAO in the memoization stores to break cycles.
        """
        super().register(source_object, dao_instance)
        self.keep_alive[id(source_object)] = source_object


@dataclass
class FromDataAccessObjectWorkItem(DataAccessObjectWorkItem):
    """
    Work item for converting a Data Access Object back to a domain object.
    """

    domain_object: Any


@dataclass
class FromDataAccessObjectState(DataAccessObjectState[FromDataAccessObjectWorkItem]):
    """
    Encapsulates the conversion state for from_dao conversions.

    Bundles memoization and in-progress tracking and provides helpers for
    allocation, circular detection, and fix-ups.
    """

    in_progress: InProgressDict = field(default_factory=dict)
    """
    Dictionary that marks objects as currently being processed by the `from_dao` method.
    """

    def add_to_queue(self, dao_instance: DataAccessObject, domain_object: Any):
        """
        Add a new work item to the processing queue.
        """
        self.queue.append(
            FromDataAccessObjectWorkItem(
                dao_instance=dao_instance, domain_object=domain_object
            )
        )

    def allocate_and_memoize(
        self, dao_instance: DataAccessObject, original_clazz: Type
    ) -> Any:
        """
        Allocates a new instance of the specified class and stores it in a memoization
        dictionary to avoid duplicating object construction for the same identifier.
        """
        result = original_clazz.__new__(original_clazz)
        self.register(dao_instance, result)
        self.in_progress[id(dao_instance)] = True
        return result

    def apply_circular_fixes(
        self, domain_object: Any, circular_references: Dict[str, Any]
    ) -> None:
        """
        Fixes circular references in the provided `domain_object`.
        """
        for key, value in circular_references.items():
            if isinstance(value, list):
                fixed_list = [self.get(v) for v in value]
                setattr(domain_object, key, fixed_list)
            else:
                setattr(domain_object, key, self.get(value))


class HasGeneric(Generic[T]):

    @classmethod
    @lru_cache(maxsize=None)
    def original_class(cls) -> T:
        """
        :return: The concrete generic argument for DAO-like bases.
        :raises NoGenericError: If no generic argument is found.
        """
        tp = cls._dao_like_argument()
        if tp is None:
            raise NoGenericError(cls)
        return tp

    @classmethod
    def _dao_like_argument(cls) -> Optional[Type]:
        """
        :return: The concrete generic argument for DAO-like bases.
        """
        # filter for instances of generic aliases in the superclasses
        for base in filter(
            lambda x: isinstance(x, _GenericAlias),
            cls.__orig_bases__,
        ):
            return get_args(base)[0]

        # No acceptable base found
        return None


class DataAccessObject(HasGeneric[T]):
    """
    This class defines the interfaces the DAO classes should implement.

    ORMatic generates classes from your python code that are derived from the provided classes in your package.
    The generated classes can be instantiated from objects of the given classes and vice versa.
    This class implements the necessary functionality.
    """

    def __init__(self, *args, **kwargs):
        """
        Allow constructing DAO instances with positional arguments that map to
        data columns.
        """
        if args and not kwargs:
            positional_kwargs = self._map_positional_arguments_to_data_columns(args)
            if positional_kwargs:
                super().__init__(**positional_kwargs)
                return
        super().__init__(*args, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init = getattr(cls, "__init__", None)

        def init_with_positional(self, *args, **kw):
            if args and not kw:
                positional_kwargs = self._map_positional_arguments_to_data_columns(args)
                if positional_kwargs:
                    if original_init:
                        return original_init(self, **positional_kwargs)
                    return super(cls, self).__init__(**positional_kwargs)
            if original_init:
                return original_init(self, *args, **kw)
            return super(cls, self).__init__(*args, **kw)

        # Inject only if the class did not already define a positional-friendly constructor
        cls.__init__ = init_with_positional

    def _map_positional_arguments_to_data_columns(
        self, arguments: Tuple[Any, ...]
    ) -> Optional[Dict[str, Any]]:
        """
        Map positional arguments to data columns if the number of arguments matches.
        """
        try:
            mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(type(self))
            data_columns = [
                column for column in mapper.columns if is_data_column(column)
            ]
            if len(arguments) == len(data_columns):
                return {
                    column.name: value for column, value in zip(data_columns, arguments)
                }
        except Exception:
            # If inspection fails or mapping is not aligned, fall back
            pass
        return None

    @classmethod
    def to_dao(
        cls,
        source_object: T,
        state: Optional[ToDataAccessObjectState] = None,
        register=True,
    ) -> _DAO:
        """
        Convert an object to its Data Access Object.

        Ensures memoization to prevent duplicate work, applies alternative
        mappings when needed, and delegates to the appropriate conversion
        strategy based on inheritance.

        :param source_object: Object to be converted into its DAO equivalent
        :param state: The state to use as context
        :param register: Whether to register the DAO class in the memo.
        :return: Instance of the DAO class (_DAO) that represents the input object after conversion
        """

        state = state or ToDataAccessObjectState()

        # check if this object has been build already
        existing = state.get(source_object)
        if existing is not None:
            return existing

        dao_source_object = state.apply_alternative_mapping_if_needed(
            cls, source_object
        )
        # If alternative mapping returned an object of a different type that is not a DAO,
        # it might be the case for recursive AlternativeMappings.
        # But if it's already a DAO (of this class or a subclass), we should use it.
        if isinstance(dao_source_object, cls):
            if register:
                state.register(source_object, dao_source_object)
            return dao_source_object

        # Determine the appropriate DAO base to consider for alternative mappings.
        alternative_base: Optional[Type[DataAccessObject]] = None
        for base_clazz in cls.__mro__[1:]:  # skip cls itself
            try:
                if issubclass(base_clazz, DataAccessObject) and issubclass(
                    base_clazz.original_class(), AlternativeMapping
                ):
                    alternative_base = base_clazz
                    break
            except Exception:
                # Some bases may not be DAOs or may not have generic info; skip safely
                continue

        result = cls()

        if register:
            state.register(source_object, result)
            if id(source_object) != id(dao_source_object):
                state.register(dao_source_object, result)

        # Add to queue for attribute/relationship filling
        is_entry_call = len(state.queue) == 0
        state.add_to_queue(dao_source_object, result, alternative_base)

        if is_entry_call:
            while state.queue:
                work_item = state.queue.popleft()
                if work_item.alternative_base is not None:
                    work_item.dao_instance.fill_dao_if_subclass_of_alternative_mapping(
                        source_object=work_item.source_object,
                        alternative_base=work_item.alternative_base,
                        state=state,
                    )
                else:
                    work_item.dao_instance.fill_dao_default(
                        source_object=work_item.source_object, state=state
                    )

        return result

    @classmethod
    def uses_alternative_mapping(cls, class_to_check: Type) -> bool:
        """
        :param class_to_check: The class to check
        :return: If the class to check uses an alternative mapping to specify the DAO or not.
        """
        return issubclass(class_to_check, DataAccessObject) and issubclass(
            class_to_check.original_class(), AlternativeMapping
        )

    @classmethod
    def _find_alternative_mapping_base(cls) -> Optional[Type[DataAccessObject]]:
        """
        Find the first base class that uses an alternative mapping.
        """
        for base_clazz in cls.__mro__[1:]:
            try:
                if issubclass(base_clazz, DataAccessObject) and issubclass(
                    base_clazz.original_class(), AlternativeMapping
                ):
                    return base_clazz
            except (AttributeError, TypeError, NoGenericError):
                continue
        return None

    def fill_dao_default(
        self, source_object: T, state: ToDataAccessObjectState
    ) -> None:
        """
        Populate this DAO instance with data from the given object.
        """
        mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(type(self))

        self.get_columns_from(source_object=source_object, columns=mapper.columns)
        self.fill_relationships_from(
            source_object=source_object,
            relationships=mapper.relationships,
            state=state,
        )

    def fill_dao_if_subclass_of_alternative_mapping(
        self,
        source_object: T,
        alternative_base: Type[DataAccessObject],
        state: ToDataAccessObjectState,
    ) -> None:
        """
        Populate this DAO instance if it is a subclass of an alternatively mapped entity.
        """
        # Temporarily remove the object from the memo to allow the parent DAO to be created separately
        temp_dao = state.pop(source_object)

        # create dao of alternatively mapped superclass
        parent_dao = alternative_base.original_class().to_dao(source_object, state)

        # Restore the object in the memo dictionary
        if temp_dao is not None:
            state.register(source_object, temp_dao)

        mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(type(self))
        parent_mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(
            alternative_base
        )

        # Split columns into those from parent and those from this DAO's table
        columns_of_parent = parent_mapper.columns
        parent_column_names = {c.name for c in columns_of_parent}
        columns_of_this_table = [
            c for c in mapper.columns if c.name not in parent_column_names
        ]

        # Copy values from parent DAO and original object
        self.get_columns_from(parent_dao, columns_of_parent)
        self.get_columns_from(source_object, columns_of_this_table)

        # Ensure columns on intermediate ancestors are also covered
        for prop in mapper.column_attrs:
            if prop.key not in parent_column_names:
                try:
                    col = prop.columns[0]
                    if is_data_column(col):
                        setattr(self, prop.key, getattr(source_object, prop.key))
                except (IndexError, AttributeError):
                    continue

        # Partition and fill relationships
        relationships_of_parent, relationships_of_this_table = (
            self._partition_parent_child_relationships(parent_mapper, mapper)
        )
        self.fill_relationships_from(parent_dao, relationships_of_parent, state)
        self.fill_relationships_from(source_object, relationships_of_this_table, state)

    def _partition_parent_child_relationships(
        self, parent: sqlalchemy.orm.Mapper, child: sqlalchemy.orm.Mapper
    ) -> Tuple[
        List[RelationshipProperty[Any]],
        List[RelationshipProperty[Any]],
    ]:
        """
        Partition the relationships by parent-only and child-only relationships.
        """
        parent_rel_keys = {rel.key for rel in parent.relationships}
        relationships_of_parent = parent.relationships
        relationships_of_child = [
            rel for rel in child.relationships if rel.key not in parent_rel_keys
        ]
        return relationships_of_parent, relationships_of_child

    def get_columns_from(self, source_object: Any, columns: Iterable[Column]) -> None:
        """
        Retrieves and assigns values from specified columns of a given object.
        """
        for column in columns:
            if is_data_column(column):
                setattr(self, column.name, getattr(source_object, column.name))

    def fill_relationships_from(
        self,
        source_object: Any,
        relationships: Iterable[RelationshipProperty],
        state: ToDataAccessObjectState,
    ) -> None:
        """
        Retrieve and update relationships from an object.
        """
        for relationship in relationships:
            is_single = relationship.direction == MANYTOONE or (
                relationship.direction == ONETOMANY and not relationship.uselist
            )
            if is_single:
                self._extract_single_relationship(source_object, relationship, state)
            elif relationship.direction in (ONETOMANY, MANYTOMANY):
                self._extract_collection_relationship(
                    source_object, relationship, state
                )

    def _extract_single_relationship(
        self,
        source_object: Any,
        relationship: RelationshipProperty,
        state: ToDataAccessObjectState,
    ) -> None:
        """
        Extract a single-valued relationship and assign the corresponding DAO.
        """
        value = getattr(source_object, relationship.key)
        if value is None:
            setattr(self, relationship.key, None)
            return

        dao_instance = self._get_or_queue_dao(value, state)
        setattr(self, relationship.key, dao_instance)

    def _extract_collection_relationship(
        self,
        source_object: Any,
        relationship: RelationshipProperty,
        state: ToDataAccessObjectState,
    ) -> None:
        """
        Extract a collection-valued relationship and assign a list of DAOs.
        """
        source_collection = getattr(source_object, relationship.key)
        dao_collection = [self._get_or_queue_dao(v, state) for v in source_collection]
        setattr(self, relationship.key, type(source_collection)(dao_collection))

    def _get_or_queue_dao(
        self, source_object: Any, state: ToDataAccessObjectState
    ) -> DataAccessObject:
        """
        Ensure a DAO exists for the given object and queue it for processing if new.
        """
        # Check if already built
        existing = state.get(source_object)
        if existing is not None:
            return existing

        dao_clazz = get_dao_class(type(source_object))
        if dao_clazz is None:
            raise NoDAOFoundDuringParsingError(source_object, type(self))

        # Check for alternative mapping
        mapped_object = state.apply_alternative_mapping_if_needed(
            dao_clazz, source_object
        )
        if isinstance(mapped_object, dao_clazz):
            state.register(source_object, mapped_object)
            return mapped_object

        # Create new DAO instance
        result = dao_clazz()
        state.register(source_object, result)
        if id(source_object) != id(mapped_object):
            state.register(mapped_object, result)

        # Queue for filling
        alternative_base = dao_clazz._find_alternative_mapping_base()
        state.add_to_queue(mapped_object, result, alternative_base)

        return result

    def from_dao(
        self,
        state: Optional[FromDataAccessObjectState] = None,
    ) -> T:
        """
        Convert this Data Access Object into its domain model instance.

        Uses a two-phase approach: allocate and memoize first to break cycles,
        then populate scalars and relationships, handle alternative mapping
        inheritance, initialize, and finally fix circular references.
        """
        state = state or FromDataAccessObjectState()

        if state.has(self):
            return state.get(self)

        result = self._allocate_uninitialized_and_memoize(state)

        # Add to queue for processing
        is_entry_call = len(state.queue) == 0
        state.add_to_queue(self, result)

        if is_entry_call:
            while state.queue:
                work_item = state.queue.popleft()
                work_item.dao_instance._fill_from_dao(work_item.domain_object, state)

            # After processing all, remove from in_progress
            state.in_progress.clear()

        return state.get(self)

    def _fill_from_dao(self, domain_object: T, state: FromDataAccessObjectState) -> T:
        """
        Actually fill the domain object with data from this DAO.
        """
        mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(type(self))

        argument_names = self._argument_names()
        scalar_keyword_arguments = self._collect_scalar_keyword_arguments(
            mapper, argument_names
        )

        relationship_keyword_arguments, circular_references = (
            self._collect_relationship_keyword_arguments(mapper, argument_names, state)
        )
        keyword_arguments = {
            **scalar_keyword_arguments,
            **relationship_keyword_arguments,
        }

        base_keyword_arguments = (
            self._build_base_keyword_arguments_for_alternative_parent(
                argument_names, state
            )
        )

        init_arguments = {**base_keyword_arguments, **keyword_arguments}
        self._call_initializer_or_assign(domain_object, init_arguments)

        # After __init__, populate remaining relationships that were not in argument_names
        self._populate_remaining_relationships(
            domain_object, mapper, argument_names, state
        )

        state.apply_circular_fixes(domain_object, circular_references)

        if isinstance(domain_object, AlternativeMapping):
            final_result = domain_object.create_from_dao()
            # Update memo if AlternativeMapping changed the instance
            state.register(self, final_result)
            return final_result

        return domain_object

    def _populate_remaining_relationships(
        self,
        domain_object: T,
        mapper: sqlalchemy.orm.Mapper,
        argument_names: List[str],
        state: FromDataAccessObjectState,
    ) -> None:
        """
        Populate relationships that were not provided to the constructor.
        """
        all_relationship_keys = {rel.key for rel in mapper.relationships}
        remaining_relationship_keys = all_relationship_keys - set(argument_names)

        for relationship in mapper.relationships:
            if relationship.key not in remaining_relationship_keys:
                continue

            value = getattr(self, relationship.key)
            is_single = relationship.direction == MANYTOONE or (
                relationship.direction == ONETOMANY and not relationship.uselist
            )

            if is_single:
                if value is None:
                    setattr(domain_object, relationship.key, None)
                    continue
                instance = self._get_or_allocate_domain_object(value, state)
                setattr(domain_object, relationship.key, instance)
            elif relationship.direction in (ONETOMANY, MANYTOMANY):
                if not value:
                    setattr(domain_object, relationship.key, value)
                    continue
                instances = [
                    self._get_or_allocate_domain_object(v, state) for v in value
                ]
                setattr(domain_object, relationship.key, type(value)(instances))

    def _get_or_allocate_domain_object(
        self, dao_instance: DataAccessObject, state: FromDataAccessObjectState
    ) -> Any:
        """
        Ensure a domain object exists for the given DAO and queue it for processing if new.
        """
        if state.has(dao_instance):
            return state.get(dao_instance)

        instance = dao_instance._allocate_uninitialized_and_memoize(state)
        state.add_to_queue(dao_instance, instance)
        return instance

    def _allocate_uninitialized_and_memoize(
        self, state: FromDataAccessObjectState
    ) -> Any:
        """
        Allocate an uninitialized domain object and memoize immediately.
        """
        return state.allocate_and_memoize(self, self.original_class())

    def _argument_names(self) -> List[str]:
        """
        :return: __init__ argument names of the original class (excluding self).
        """
        init_of_original_class = self.original_class().__init__
        return [
            parameter.name
            for parameter in inspect.signature(
                init_of_original_class
            ).parameters.values()
        ][1:]

    def _collect_scalar_keyword_arguments(
        self, mapper: sqlalchemy.orm.Mapper, argument_names: List[str]
    ) -> Dict[str, Any]:
        """
        :return: keyword arguments for scalar columns present in the constructor.
        """
        keyword_arguments: Dict[str, Any] = {}
        for column in mapper.columns:
            if column.name in argument_names and is_data_column(column):
                keyword_arguments[column.name] = getattr(self, column.name)
        return keyword_arguments

    def _collect_relationship_keyword_arguments(
        self,
        mapper: sqlalchemy.orm.Mapper,
        argument_names: List[str],
        state: FromDataAccessObjectState,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Collect relationship constructor arguments and capture circular references.
        """
        relationship_keyword_arguments: Dict[str, Any] = {}
        circular_references: Dict[str, Any] = {}

        for relationship in mapper.relationships:
            if relationship.key not in argument_names:
                continue

            value = getattr(self, relationship.key)
            is_single = relationship.direction == MANYTOONE or (
                relationship.direction == ONETOMANY and not relationship.uselist
            )

            if is_single:
                if value is None:
                    relationship_keyword_arguments[relationship.key] = None
                    continue

                if state.has(value):
                    parsed = state.get(value)
                    if id(value) in state.in_progress:
                        circular_references[relationship.key] = value
                    relationship_keyword_arguments[relationship.key] = parsed
                else:
                    original_clazz = value.original_class()
                    if issubclass(original_clazz, AlternativeMapping):
                        parsed = value.from_dao(state=state)
                    else:
                        parsed = value._allocate_uninitialized_and_memoize(state)
                        state.add_to_queue(value, parsed)
                        circular_references[relationship.key] = value
                    relationship_keyword_arguments[relationship.key] = parsed
            elif relationship.direction in (ONETOMANY, MANYTOMANY):
                if not value:
                    relationship_keyword_arguments[relationship.key] = value
                    continue

                instances = []
                circular_values: List[Any] = []
                for v in value:
                    if state.has(v):
                        instance = state.get(v)
                        if id(v) in state.in_progress:
                            circular_values.append(v)
                        instances.append(instance)
                    else:
                        original_clazz = v.original_class()
                        if issubclass(original_clazz, AlternativeMapping):
                            instance = v.from_dao(state=state)
                        else:
                            instance = v._allocate_uninitialized_and_memoize(state)
                            state.add_to_queue(v, instance)
                            circular_values.append(v)
                        instances.append(instance)

                if circular_values:
                    circular_references[relationship.key] = circular_values
                relationship_keyword_arguments[relationship.key] = type(value)(
                    instances
                )
            else:
                raise UnsupportedRelationshipError(relationship)

        return relationship_keyword_arguments, circular_references

    def _build_base_keyword_arguments_for_alternative_parent(
        self,
        argument_names: List[str],
        state: FromDataAccessObjectState,
    ) -> Dict[str, Any]:
        """
        Build keyword arguments for an alternative parent.
        """
        base_clazz = self.__class__.__bases__[0]
        base_keyword_arguments: Dict[str, Any] = {}
        if self.uses_alternative_mapping(base_clazz):
            parent_dao = base_clazz()
            parent_mapper = sqlalchemy.inspection.inspect(base_clazz)
            for column in parent_mapper.columns:
                if is_data_column(column):
                    setattr(parent_dao, column.name, getattr(self, column.name))
            for relationship in parent_mapper.relationships:
                setattr(parent_dao, relationship.key, getattr(self, relationship.key))
            base_result = parent_dao.from_dao(state=state)
            for argument in argument_names:
                if argument not in base_keyword_arguments and not hasattr(
                    self, argument
                ):
                    try:
                        base_keyword_arguments[argument] = getattr(
                            base_result, argument
                        )
                    except AttributeError:
                        continue
        return base_keyword_arguments

    @classmethod
    def _call_initializer_or_assign(
        cls, result: Any, init_args: Dict[str, Any]
    ) -> None:
        """
        Call the original __init__. If it fails due to signature mismatch, assign attributes directly.
        """
        try:
            result.__init__(**init_args)
        except TypeError as e:
            logging.getLogger(__name__).debug(
                f"from_dao __init__ call failed with {e}; falling back to manual assignment. "
                f"This might skip side effects of the original initialization."
            )
            for key, val in init_args.items():
                setattr(result, key, val)

    @classmethod
    def _apply_circular_fixes(
        cls, result: Any, circular_refs: Dict[str, Any], state: "FromDAOState"
    ) -> None:
        """
        Replace circular placeholder DAOs with their finalized domain objects using the state.
        """
        state.apply_circular_fixes(result, circular_refs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        try:
            mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(type(self))
            # Compare only data columns, ignoring PK/FK/polymorphic columns
            return all(
                getattr(self, column.name) == getattr(other, column.name)
                for column in mapper.columns
                if is_data_column(column)
            )
        except Exception:
            # Fallback to identity comparison if we cannot inspect
            return self is other

    def __repr__(self) -> str:
        if not hasattr(_repr_thread_local, "seen"):
            _repr_thread_local.seen = set()

        if id(self) in _repr_thread_local.seen:
            return f"{self.__class__.__name__}(...)"

        _repr_thread_local.seen.add(id(self))
        try:
            mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(type(self))
            representations = []

            for column in mapper.columns:
                if is_data_column(column):
                    value = getattr(self, column.name)
                    representations.append(f"{column.name}={repr(value)}")

            for relationship in mapper.relationships:
                value = getattr(self, relationship.key)
                representations.append(f"{relationship.key}={repr(value)}")

            return f"{self.__class__.__name__}({', '.join(representations)})"
        finally:
            _repr_thread_local.seen.remove(id(self))


class AlternativeMapping(HasGeneric[T], abc.ABC):

    @classmethod
    def to_dao(
        cls, source_object: T, state: Optional[ToDataAccessObjectState] = None
    ) -> _DAO:
        """
        Create a DAO from the source_object if it doesn't exist.
        """
        state = state or ToDataAccessObjectState()
        if state.has(source_object):
            return state.get(source_object)
        elif isinstance(source_object, cls):
            return source_object
        else:
            result = cls.create_instance(source_object)
            return result

    @classmethod
    @abc.abstractmethod
    def create_instance(cls, obj: T) -> Self:
        """
        Create a DAO from the obj.
        The method needs to be overloaded by the user.

        :param obj: The obj to create the DAO from.
        :return: An instance of this class created from the obj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def create_from_dao(self) -> T:
        """
        Creates an object from a Data Access Object (DAO) by using the predefined
        logic and transformations specific to the implementation. This facilitates
        constructing domain-specific objects from underlying data representations.

        :return: The object created from the DAO.
        :rtype: T
        """
        raise NotImplementedError


@lru_cache(maxsize=None)
def _get_clazz_by_original_clazz(
    base_clazz: Type, original_clazz: Type
) -> Optional[Type]:
    """
    Find a subclass of base_clazz that maps to original_clazz.
    """
    for subclass in recursive_subclasses(base_clazz):
        try:
            if subclass.original_class() == original_clazz:
                return subclass
        except (AttributeError, TypeError, NoGenericError):
            continue
    return None


@lru_cache(maxsize=None)
def get_dao_class(original_clazz: Type) -> Optional[Type[DataAccessObject]]:
    """
    Find the DAO class for the given original class.
    """
    alternative_mapping = get_alternative_mapping(original_clazz)
    if alternative_mapping is not None:
        original_clazz = alternative_mapping

    return _get_clazz_by_original_clazz(DataAccessObject, original_clazz)


@lru_cache(maxsize=None)
def get_alternative_mapping(
    original_clazz: Type,
) -> Optional[Type[AlternativeMapping]]:
    """
    Find the alternative mapping for the given original class.
    """
    return _get_clazz_by_original_clazz(AlternativeMapping, original_clazz)


def to_dao(
    source_object: Any, state: Optional[ToDataAccessObjectState] = None
) -> DataAccessObject:
    """
    Convert any object to a dao class.
    """
    dao_clazz = get_dao_class(type(source_object))
    if dao_clazz is None:
        raise NoDAOFoundError(source_object)
    state = state or ToDataAccessObjectState()
    return dao_clazz.to_dao(source_object, state)


# Compatibility aliases for backward compatibility and to avoid breaking existing tests.
ToDAOState = ToDataAccessObjectState
FromDAOState = FromDataAccessObjectState
