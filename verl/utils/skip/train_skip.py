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

from typing import Callable

from verl.protocol import DataProto
from verl.utils.skip.base_skip import BaseSkip, SkipAction, register_skip


@register_skip("train")
class TrainSkip(BaseSkip):
    """TrainSkip skips _update_critic and _update_actor calls at specified steps."""

    support_actions = [SkipAction.EMPTY]
    print_mark = "[TrainSkip()] "

    def meet_precondition(self) -> bool:
        return True

    def warp_function(self, func: Callable, *args, **kwargs):
        print(
            f"{self.print_mark}Skip _update_critic/_update_actor at step {self.global_step}",
            flush=True,
        )
        return DataProto.from_single_dict({}, meta_info={"metrics": {}})

    def prepare_data(self, result, *args, **kwargs):
        pass  # no-op for empty action
