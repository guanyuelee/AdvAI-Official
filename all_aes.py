# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All auto-encoder models."""

import baseline, denoising, dropout, aae, vae, midrae, vqvae, mi_acai,  dr_acai, acai, giacai, gidrae2, \
    gilracai_all8, gilracai_all9, gidrvatae, absacai, absmiae, gmivatdrae_independent, gmidrae_independent, \
    absgmiacai_independent, absgmilracai2_independent, absgiacai, lrmi_acai, gmiacai, \
    gmiacai_independent, gmilracai_independent, absgmilracai_independent, vaegan

ALL_AES = {x.__name__: x for x in
           [baseline.AEBaseline, denoising.AEDenoising, dropout.AEDropout, mi_acai.MIAE,
            aae.AAE, vae.VAE, vqvae.AEVQVAE, midrae.MIDRAE, acai.ACAI,  dr_acai.DRACAI,
            giacai.GIACAI, gidrae2.GIDRAE2, gilracai_all8.GILRACAI8, gilracai_all9.GILRACAI9,
            gidrvatae.GIDRAEVAT, absacai.ABSACAI, absmiae.ABSMIAE, gmivatdrae_independent.GMINDVATDRAE,
            gmidrae_independent.GMINDDRAE, absgmiacai_independent.ABSGMINDACAI,
            absgmilracai2_independent.ABSGMINDLRACAI2, absgiacai.ABSGIACAI,
            lrmi_acai.MILRAE, gmiacai.GMIACAI, gmiacai_independent.GMINDACAI,
            gmilracai_independent.GMINDLRACAI, absgmilracai_independent.ABSGMINDLRACAI,
            vaegan.VAEGAN]}

