/** Sample paragraphs spanning the registers the model was trained on, plus a few it wasn't. */
export interface Sample {
  label: string;
  text: string;
}

export const SAMPLES: Sample[] = [
  {
    label: 'slack',
    text: `Hey team, heads up: the staging deploy failed twice this morning because the migration lock wasn't released. I've rolled back to build 412 and we are NOT shipping today. Ravi is looking into the lock cleanup; I'll post an update by 4pm. If you need staging urgently, ping me directly rather than retrying the deploy — each retry takes another lock and makes the cleanup slower.`,
  },
  {
    label: 'news',
    text: `The James Webb Space Telescope has detected carbon dioxide in the atmosphere of a planet orbiting another star, the first unambiguous detection of the gas outside our solar system. The planet, WASP-39b, is a hot gas giant roughly the mass of Saturn but with a diameter 1.3 times larger than Jupiter, and it orbits its host star every four days. Researchers say the finding demonstrates that the telescope can characterise the atmospheres of smaller, rocky worlds in the years ahead.`,
  },
  {
    label: 'contract',
    text: `The tenant shall not sublet the premises without the prior written consent of the landlord, which consent shall not be unreasonably withheld. Any purported sublease made without such consent is void. The tenant remains liable for all obligations under this agreement notwithstanding any permitted sublease, and no sublease shall extend beyond the term of this agreement.`,
  },
  {
    label: 'paper',
    text: `Exposure bias describes the phenomenon that a language model trained under the teacher forcing schema may perform poorly at the inference stage when its predictions are conditioned on its previous predictions unseen from the training corpus. Recently, several generative adversarial networks and reinforcement learning methods have been introduced to alleviate this problem. Nonetheless, a common issue in these approaches is the sparsity of reward signals, which slows convergence and increases variance.`,
  },
  {
    label: 'docs',
    text: `By default the client retries idempotent requests up to three times with exponential backoff, starting at 100 milliseconds. Non-idempotent requests are never retried automatically, because the server may have already applied the change. To override this, pass a retry policy to the constructor; passing null disables retries entirely. Note that the timeout applies to each attempt individually, not to the whole sequence.`,
  },
  {
    label: 'recipe',
    text: `Heat the oil in a wide pan over medium heat, then add the onions and cook for about 8 minutes until soft but not browned. Stir in the garlic and cook for another minute. Add the tomatoes, reduce the heat, and simmer uncovered for 25 minutes until the sauce thickens noticeably. Season with salt just before serving, since salting early draws water out of the tomatoes and thins the sauce.`,
  },
];
