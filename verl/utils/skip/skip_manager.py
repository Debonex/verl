# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import inspect
from typing import Any, Callable, Optional

from verl.utils.skip.base_skip import BaseSkip, SKIP_REGISTRY

  # noqa: F401  # side effect: register "rollout" in SKIP_REGISTRY


class SkipManager:
    skip_instances: dict[str, BaseSkip]
    config: Any
    step: int

    @classmethod
    def init(cls, config: Any):
        cls.config = config
        cls.step = -1
        cls.skip_instances = {}
        for name, skip_cls in SKIP_REGISTRY.items():
            instance = skip_cls(cls.config.skip.get(name), cls.config)
            cls.skip_instances[name] = instance
            print(
                    f"\033[33mSkip instance {name} initialized with config: {cls.config.skip.get(name)}\033[0m",
                    flush=True,
                )

    @classmethod
    def set_step(cls, step: int):
        cls.step = step

    @classmethod
    def annotate(cls, role: str, **kwargs_outer) -> Callable:
        def decorator(func: Callable) -> Callable:
            if inspect.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs_inner):
                    skip_instance = cls.skip_instances[role]
                    if not skip_instance.is_enabled() or cls.step not in skip_instance.steps:
                        return await func(*args, **kwargs_inner)
                    skip_instance.set_context(cls.step)
                    if skip_instance.meet_precondition():
                        return skip_instance.warp_function(func, *args, **kwargs_inner)
                    result = await func(*args, **kwargs_inner)
                    skip_instance.prepare_data(result, *args, **kwargs_inner)
                    return result

                return async_wrapper

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs_inner):
                skip_instance = cls.skip_instances[role]
                if not skip_instance.is_enabled() or cls.step not in skip_instance.steps:
                    return func(*args, **kwargs_inner)
                skip_instance.set_context(cls.step)
                if skip_instance.meet_precondition():
                    return skip_instance.warp_function(func, *args, **kwargs_inner)
                result = func(*args, **kwargs_inner)
                skip_instance.prepare_data(result, *args, **kwargs_inner)
                return result

            return sync_wrapper

        return decorator
