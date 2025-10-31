import torch
import torch.nn.functional as F
from models.ode_transformer_gpt import CenterNorm
from torchvision.transforms.functional import gaussian_blur
import math
from collections import defaultdict
from typing import Optional

from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)


class TemperatureScheduler:
    def __init__(self, initial_temp=6.0, final_temp=1.5, total_epochs=100):
        self.init_t = initial_temp
        self.final_t = final_temp
        self.total_epochs = total_epochs

    def get_temp(self, epoch):
        # Cosine-style decay is smoother than linear
        ratio = epoch / self.total_epochs
        current_t = self.final_t + 0.5 * (self.init_t - self.final_t) * (
            1 + math.cos(math.pi * ratio)
        )
        return current_t


class ImageDistilTrainer(torch.nn.Module):
    def __init__(
        self,
        teacher_model=None,
        student_model=None,
        optimizer=None,
        scheduler=None,
        mse_full_path: bool = False,
        use_distillation: bool = True,
        use_supervision: bool = True,
        use_mse_loss: bool = True,
        temperature=None,
        jasmin_k: int = 10,
        lambda_param=None,
        curriculum: bool = False,
        patience_factor: int = 0.1,
    ):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.loss_function = torch.nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = torch.nn.MSELoss(reduction="none")
        self.L1_loss = torch.nn.L1Loss(reduction="none")
        self.teacher.eval()
        self.student.train()

        self.temperature = temperature
        self.lambda_param = lambda_param
        self.mse_loss_full_path = mse_full_path
        self.use_mse_loss = use_mse_loss
        self.use_distillation = use_distillation
        self.use_supervision = use_supervision
        self.jasmin_k = jasmin_k
        self.patience_factor = patience_factor
        self.optimizer = optimizer

        self.temperature_scheduler = TemperatureScheduler(
            initial_temp=temperature, final_temp=1.0, total_epochs=300
        )

        self.alpha_param = 0.01
        self.representation_checkpoint = None
        self.train_class = False

        self.scheduler = scheduler

        self.curriculum = curriculum

    def extract_mass(
        self,
        attentions_last_head,
        threshold=0.8,
        patch_size: int = 16,
        smooth=True,
        scale_factor=40,
        return_mask: bool = False,
    ):
        B, nh, num_patches = attentions_last_head.shape
        h_featmap = w_featmap = int(num_patches**0.5 + 0.5)

        val, idx = torch.sort(attentions_last_head, dim=-1)
        val = val / (val.sum(dim=(-1), keepdim=True) + 1e-8)
        cumval = torch.cumsum(val, dim=-1)

        if smooth:
            mask_soft = torch.sigmoid((cumval - (1 - threshold)) * scale_factor)
        else:
            mask_soft = (cumval > (1 - threshold)).float()

        idx2 = torch.argsort(idx, dim=-1)
        th_attn = torch.gather(mask_soft, dim=-1, index=idx2)

        th_attn = th_attn.view(B, nh, w_featmap, h_featmap).float()

        attn_reshaped = attentions_last_head.view(B, nh, w_featmap, h_featmap)

        attn_filtered = attn_reshaped * th_attn
        if smooth:
            attn_filtered = gaussian_blur(attn_filtered, kernel_size=(3, 3), sigma=0.5)

        attentions_mean = attn_filtered.mean(dim=1) * th_attn.mean(dim=1)

        if return_mask:
            return attentions_mean, attn_filtered, th_attn.mean(dim=1)
        else:
            return attentions_mean, attn_filtered, None

    def compute_mse_loss(
        self,
        student_intermediate_representations: torch.Tensor,
        teacher_intermediate_representations: torch.Tensor,
        normalize: bool = False,
    ):
        mse_losses = defaultdict(int)

        if normalize:
            teacher_intermediate_representations = F.normalize(
                teacher_intermediate_representations, p=2, dim=-1
            )
            student_intermediate_representations = F.normalize(
                student_intermediate_representations, p=2, dim=-1
            )

        if self.mse_loss_full_path:
            individual_losses = [
                self.mse_loss(t[:, 0], c[:, 0]).mean()
                for t, c in zip(
                    teacher_intermediate_representations,
                    student_intermediate_representations,
                )
            ]
            mse_losses = {
                f"mse_loss_t@{i}": loss for i, loss in enumerate(individual_losses)
            }
        else:
            individual_losses = [
                self.mse_loss(
                    teacher_intermediate_representations[-1, :, 0],
                    student_intermediate_representations[-1, :, 0],
                ).mean()
            ]
            mse_losses = {
                f"mse_loss_t@{(teacher_intermediate_representations.size(0) - 1)}": loss
                for i, loss in enumerate(individual_losses)
            }

        mse_loss = sum(individual_losses)

        return mse_loss, mse_losses

    def compute_l1_attention_loss(
        self, student_output_attentions, teacher_output_attentions
    ):
        # --- 1. Extract last-layer attention excluding CLS→CLS ---
        attn_t = torch.stack(teacher_output_attentions, dim=0)[-1][
            :, :, 0, 1:
        ]  # [B, H, N]
        attn_s = student_output_attentions[:, :, 0, 1:]  # [B, H, N]

        s_attn_mean, s_attn, _ = self.extract_mass(attn_s, threshold=0.5)
        t_attn_mean, t_attn, _ = self.extract_mass(attn_t, threshold=0.5)

        l1_loss = (self.L1_loss(s_attn_mean, t_attn_mean)).sum()

        return l1_loss * self.lambda_param

    def compute_distillation_loss(
        self,
        student_output_attentions,
        teacher_output_attentions,
        eps=1e-8,
        compute_per_head: bool = True,
    ):
        ## TODO Experiment with the attention distillation by heads.
        """
        Computes symmetrized, temperature-scaled attention distillation loss.

        Args:
            teacher_output_attentions: torch.Tensor or List [L, B, H, N, N]
            student_output_attentions: torch.Tensor or List [L, B, H, N, N]
        """

        # --- 1. Extract last-layer attention excluding CLS→CLS ---
        attn_t = torch.stack(teacher_output_attentions, dim=0)[-1][
            :, :, 0, 1:
        ]  # [B, H, N]
        attn_s = student_output_attentions[:, :, 0, 1:]  # [B, H, N]

        s_attn_mean, s_attn, _ = self.extract_mass(attn_s, threshold=0.5)
        t_attn_mean, t_attn, _ = self.extract_mass(attn_t, threshold=0.5)

        if not compute_per_head:
            s_attn_mean = s_attn_mean.clamp(min=eps)
            t_attn_mean = t_attn_mean.clamp(min=eps)
            attn_s = torch.log(s_attn_mean + eps).sum(dim=1)  # [B, N]
            attn_t = torch.log(t_attn_mean + eps).sum(dim=1)  # [B, N]

            temp = getattr(self, "temperature", 1.0)
            attn_s = F.log_softmax(attn_s / temp, dim=-1)
            attn_t = F.softmax(attn_t / temp, dim=-1)

            kl_st = F.kl_div(attn_s, attn_t, reduction="batchmean")  # Student→Teacher
            kl_ts = F.kl_div(
                attn_t.log(), attn_s.exp(), reduction="batchmean"
            )  # Teacher→Student

            sym_kl = 0.5 * (kl_st + kl_ts) * (temp**2)

            total_loss = sym_kl

        else:
            attn_s = torch.log(s_attn + eps).sum(dim=3)  # [B, H, N]
            attn_t = torch.log(t_attn + eps).sum(dim=3)  # [B, H, N]

            temp = getattr(self, "temperature", 1.0)
            attn_s = F.log_softmax(attn_s / temp, dim=2)  # [B, H, N]
            attn_t = F.softmax(attn_t / temp, dim=2)  # [B, H, N]

            kl_st = F.kl_div(attn_s, attn_t, reduction="none")  # [B, H, N]
            kl_st = kl_st.sum(dim=2).mean(dim=0)  # [H] (mean over mini-batch)

            # For symmetric KL:
            kl_ts = F.kl_div(attn_t.log(), attn_s.exp(), reduction="none")  # [B, H, N]
            kl_ts = kl_ts.sum(dim=2).mean(dim=0)  # [H] (mean over mini-batch)

            sym_kl = 0.5 * (kl_st + kl_ts) * (temp**2)  # [H]

            # Optionally, sum or mean over heads for total loss:
            total_loss = sym_kl.mean()  # scalar, or keep sym_kl for per-head loss

        return total_loss

    def train_batch_representation(self, student_output, teacher_output):
        loss = 0.0

        teacher_states = torch.stack(teacher_output["hidden_states"], dim=0)[1:]

        if student_output.get("control_points", None) is None:
            control_points = student_output["states"]
            control_idx = torch.tensor(
                [
                    control_points.shape[0] / teacher_states.shape[0]
                    for i in range(teacher_states.shape[0])
                ]
            )
            checkpoints = torch.cumsum(control_idx, dim=0).long()
            checkpoints[-1] -= 1
            control_points = control_points[checkpoints]
        else:
            control_points = student_output["control_points"]

        mse_loss, mse_losses = self.compute_mse_loss(
            student_intermediate_representations=control_points,
            teacher_intermediate_representations=teacher_states,
        )

        loss += mse_loss

        dict_output = {"mse_loss": mse_loss}

        if self.use_distillation:
            self.temperature_scheduler.get_temp(epoch=self.epoch)
            kl = self.compute_l1_attention_loss(
                student_output["attentions"], teacher_output["attentions"]
            )
            if kl.isnan().any():
                print("KL loss is NaN")
            else:
                loss += kl

            dict_output.update({"kl_loss": kl})

        loss *= self.lambda_param
        dict_output.update({"loss": loss})
        dict_output.update(mse_losses)

        return dict_output

    def forward(self, inputs, labels, epoch: Optional[int] = 0):
        self.optimizer.zero_grad(set_to_none=True)

        loss = 0.0
        self.epoch = epoch
        dict_output = {}

        student_output = self.student(
            **inputs,
            labels=labels,
            output_hidden_states=True,
            output_control_points=True,
            output_attentions=True,
            jasmin_k=self.jasmin_k,
        )

        with torch.no_grad():
            teacher_output = self.teacher(
                **inputs, output_hidden_states=True, output_attentions=True
            )

        # --- Store outputs ---
        dict_output.update(
            {
                "student_output": student_output,
                "teacher_output": teacher_output,
                "second_derivative_upper_bound": student_output.get(
                    "second_derivative_upper_bound"
                ),
                "finite_difference_upper_bound": student_output.get(
                    "finite_difference_upper_bound"
                ),
            }
        )

        # --- Representation loss ---
        representation_losses = self.train_batch_representation(
            student_output, teacher_output
        )
        dict_output.update(representation_losses)
        loss += representation_losses["loss"]
        loss += student_output["jasmin_loss"]

        if self.use_supervision:
            loss += student_output["loss"]

        # --- Final checks ---
        dict_output.update(
            {
                "jasmin_loss": student_output["jasmin_loss"],
                "supervision_loss": student_output["loss"],
                "loss": loss,
            }
        )

        if not torch.isfinite(loss):
            print(dict_output)
            raise ValueError("Loss is NaN or Inf!")

        # --- Backpropagation ---
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 5.0)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return dict_output

    def compute_loss(self, inputs, labels, return_outputs=True):
        dict_output = {}
        losses_to_sum = 0.0
        student_output = self.student(
            **inputs,
            labels=labels,
            output_hidden_states=True,
            output_control_points=True,
            output_attention=True,
            jasmin_k=self.jasmin_k,
        )
        dict_output["student_output"] = student_output

        with torch.no_grad():
            teacher_output = self.teacher(
                **inputs, output_hidden_states=True, output_attentions=True
            )

        if self.use_mse_loss:
            if self.mse_loss_full_path:
                mse_losses = defaultdict(int)
                mse_loss = 0.0
                control_points = student_output["control_points"][:, :, 0, :]
                teacher_cls_token = torch.stack(teacher_output.hidden_states)[
                    1:, :, 0, :
                ]

                for idx in list(range(len(control_points))):  # [-2:]:
                    individual_loss = self.mse_loss(
                        teacher_cls_token[idx], control_points[idx]
                    )
                    mse_loss += (len(control_points) - idx) * individual_loss
                    mse_losses[f"mse_loss_t@{idx}"] = individual_loss

                mse_loss = mse_loss / len(control_points)

                dict_output["mse_losses"] = mse_losses

            else:
                last_cls_token = teacher_output.hidden_states[-1][:, 0]
                last_patches_tokens = teacher_output.hidden_states[-1][:, 1:]
                last_state = student_output["states"][-1]
                mse_loss_cls = self.mse_loss(last_cls_token, last_state[:, 0])
                if self.use_distillation:
                    mse_loss_patches = self.mse_loss(
                        last_patches_tokens, last_state[:, 2:]
                    )
                else:
                    mse_loss_patches = self.mse_loss(
                        last_patches_tokens, last_state[:, 1:]
                    )

                mse_loss = mse_loss_cls + (mse_loss_patches * 0.1)

            losses_to_sum += mse_loss * self.alpha_param
            dict_output["mse_loss"] = mse_loss

        if self.use_distillation:
            soft_teacher = F.softmax(
                teacher_output["logits"] / self.temperature, dim=-1
            )
            soft_student = F.log_softmax(
                student_output["logits_dist"] / self.temperature, dim=-1
            )

            # Compute the loss
            distillation_loss = self.loss_function(soft_student, soft_teacher) * (
                self.temperature**2
            )
            distillation_loss = self.lambda_param * distillation_loss

            losses_to_sum += distillation_loss

            dict_output["kd loss"] = distillation_loss

        if self.use_supervision:
            student_target_loss = student_output["loss"] * (1 - self.lambda_param)
            losses_to_sum += student_target_loss

            dict_output["student_target_loss"] = student_target_loss

        dict_output["loss"] = losses_to_sum

        return dict_output
