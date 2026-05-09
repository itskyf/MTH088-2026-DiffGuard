"""Generate 100 Z-Image Turbo images for AI-generated image detection."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any

import fal_client
import httpx
import typer

MODEL_ID = "fal-ai/z-image/turbo"

IMAGE_DIR = Path("data/bronze/z_image_turbo")
METADATA_PATH = IMAGE_DIR / "metadata.csv"

NUM_INFERENCE_STEPS = 8
OUTPUT_FORMAT = "png"
ENABLE_PROMPT_EXPANSION = False
ACCELERATION = "regular"
NUM_IMAGES = 1
SYNC_MODE = False
ENABLE_SAFETY_CHECKER = True

SEEDS = [1001, 1002, 1003, 1004, 1005]

app = typer.Typer(no_args_is_help=False)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptSpec:
    """Prompt definition used to build reproducible generation jobs."""

    prompt_id: int
    category: str
    prompt: str


@dataclass(frozen=True)
class GenerationJob:
    """Single image generation job."""

    image_id: str
    prompt_id: int
    category: str
    prompt: str
    seed: int
    image_size: str
    output_path: Path


@dataclass
class MetadataRow:
    """Metadata saved for each generated image."""

    image_id: str
    local_path: str
    model_id: str
    prompt_id: int
    category: str
    prompt: str
    seed: int
    returned_seed: Any
    image_size: str
    width: Any
    height: Any
    content_type: Any
    source_url: str
    request_id: Any
    has_nsfw_concepts: str
    timings: str
    num_inference_steps: int
    output_format: str
    enable_prompt_expansion: bool
    acceleration: str
    num_images: int
    sync_mode: bool
    enable_safety_checker: bool
    status: str
    error: str


PROMPTS: list[PromptSpec] = [
    PromptSpec(
        1,
        "outdoor_street_landscape",
        "Photorealistic daytime city street scene, motorbikes parked beside a sidewalk cafe, natural shadows, smartphone photography, realistic textures.",
    ),
    PromptSpec(
        2,
        "outdoor_street_landscape",
        "Photorealistic coastal landscape with fishing boats near a quiet harbor, cloudy morning light, natural colors, documentary travel photo.",
    ),
    PromptSpec(
        3,
        "outdoor_street_landscape",
        "Photorealistic mountain trail after light rain, wet rocks, pine trees, distant fog, handheld outdoor photography.",
    ),
    PromptSpec(
        4,
        "outdoor_street_landscape",
        "Photorealistic rainy market street, umbrellas, puddle reflections, soft overcast light, candid urban photo.",
    ),
    PromptSpec(
        5,
        "indoor_room_desk",
        "Photorealistic small apartment work desk, laptop, notebooks, coffee mug, warm window light, realistic clutter.",
    ),
    PromptSpec(
        6,
        "indoor_room_desk",
        "Photorealistic wooden kitchen interior, cutting board, ceramic bowls, morning sunlight, natural household scene.",
    ),
    PromptSpec(
        7,
        "indoor_room_desk",
        "Photorealistic quiet library reading room, wooden tables, bookshelves, soft indoor lighting, realistic perspective.",
    ),
    PromptSpec(
        8,
        "indoor_room_desk",
        "Photorealistic bedroom shelf with books, folded clothes, small lamp, neutral colors, casual indoor smartphone photo.",
    ),
    PromptSpec(
        9,
        "object_product_food_fabric",
        "Photorealistic close-up of a ceramic coffee cup on a textured table, shallow depth of field, visible small scratches.",
    ),
    PromptSpec(
        10,
        "object_product_food_fabric",
        "Photorealistic fresh fruit on a wooden cutting board, orange slices, apple, knife, kitchen counter, natural light.",
    ),
    PromptSpec(
        11,
        "object_product_food_fabric",
        "Photorealistic folded denim and cotton fabric stack, visible weave texture, soft side lighting, product photography.",
    ),
    PromptSpec(
        12,
        "object_product_food_fabric",
        "Photorealistic consumer electronics product box on a desk, matte cardboard texture, small printed icons, realistic reflections.",
    ),
    PromptSpec(
        13,
        "low_light_smartphone_motion_blur",
        "Photorealistic night street captured on smartphone, neon signs, mild motion blur, high ISO noise, wet pavement.",
    ),
    PromptSpec(
        14,
        "low_light_smartphone_motion_blur",
        "Photorealistic dim restaurant table scene, moving waiter in background, slight motion blur, candle light, smartphone photo.",
    ),
    PromptSpec(
        15,
        "low_light_smartphone_motion_blur",
        "Photorealistic view through a bus window at night, blurred city lights, reflections on glass, handheld smartphone capture.",
    ),
    PromptSpec(
        16,
        "low_light_smartphone_motion_blur",
        "Photorealistic indoor concert audience silhouettes, low light, stage glow, motion blur, noisy smartphone photography.",
    ),
    PromptSpec(
        17,
        "text_sign_poster_label",
        "Photorealistic local bakery storefront with a clear sign reading 'FRESH BREAD', morning street light, natural photo.",
    ),
    PromptSpec(
        18,
        "text_sign_poster_label",
        "Photorealistic community board with a poster reading 'OPEN STUDIO SATURDAY', paper texture, tape, indoor lighting.",
    ),
    PromptSpec(
        19,
        "text_sign_poster_label",
        "Photorealistic product label on a glass bottle reading 'HERBAL TEA', condensation droplets, kitchen counter.",
    ),
    PromptSpec(
        20,
        "text_sign_poster_label",
        "Photorealistic street direction sign reading 'RIVER WALK', trees in background, late afternoon light.",
    ),
]


def slugify(value: str) -> str:
    """Return a filesystem-safe slug.

    Args:
        value: Raw text to normalize.

    Returns:
        A lowercase slug using only letters, numbers, and underscores.
    """
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def deterministic_score(prompt_id: int, seed: int) -> str:
    """Return a deterministic hash for balanced variant assignment.

    Args:
        prompt_id: Prompt identifier.
        seed: Generation seed.

    Returns:
        Hex digest used for stable sorting.
    """
    key = f"{prompt_id}:{seed}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def assign_image_sizes(
    jobs: list[tuple[PromptSpec, int]],
) -> dict[tuple[int, int], str]:
    """Assign exact 70/20/10 image-size distribution.

    Args:
        jobs: Prompt and seed pairs.

    Returns:
        Mapping from ``(prompt_id, seed)`` to fal image size.
    """
    sorted_jobs = sorted(
        jobs,
        key=lambda item: deterministic_score(item[0].prompt_id, item[1]),
    )

    sizes = ["landscape_4_3"] * 70 + ["square_hd"] * 20 + ["portrait_4_3"] * 10

    return {
        (prompt_spec.prompt_id, seed): image_size
        for (prompt_spec, seed), image_size in zip(sorted_jobs, sizes, strict=True)
    }


def build_jobs() -> list[GenerationJob]:
    """Build 100 deterministic generation jobs.

    Returns:
        Generation jobs with prompt, seed, image size, and output path.
    """
    prompt_seed_pairs = [
        (prompt_spec, seed) for prompt_spec in PROMPTS for seed in SEEDS
    ]
    size_by_key = assign_image_sizes(prompt_seed_pairs)

    jobs: list[GenerationJob] = []
    for prompt_spec, seed in prompt_seed_pairs:
        image_size = size_by_key[(prompt_spec.prompt_id, seed)]
        category_slug = slugify(prompt_spec.category)
        image_id = (
            f"zimg_p{prompt_spec.prompt_id:02d}_seed{seed}_"
            f"{image_size}_steps{NUM_INFERENCE_STEPS}_{category_slug}"
        )
        output_path = IMAGE_DIR / f"{image_id}.{OUTPUT_FORMAT}"

        jobs.append(
            GenerationJob(
                image_id=image_id,
                prompt_id=prompt_spec.prompt_id,
                category=prompt_spec.category,
                prompt=prompt_spec.prompt,
                seed=seed,
                image_size=image_size,
                output_path=output_path,
            )
        )

    return jobs


def result_get(result: Any, key: str, default: Any = None) -> Any:
    """Read a key from dict-like or object-like fal results.

    Args:
        result: fal response object.
        key: Field name.
        default: Value returned when the field is missing.

    Returns:
        Extracted value.
    """
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


async def download_image(
    client: httpx.AsyncClient, url: str, output_path: Path
) -> None:
    """Download one generated image.

    Args:
        client: Shared HTTP client.
        url: Hosted image URL from fal.
        output_path: Local image path.
    """
    response = await client.get(url)
    response.raise_for_status()
    output_path.write_bytes(response.content)


async def generate_one(
    job: GenerationJob,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> MetadataRow:
    """Generate and save one image.

    Args:
        job: Single generation job.
        client: Shared HTTP client for image downloads.
        semaphore: Concurrency limiter.

    Returns:
        Metadata row for CSV output.
    """
    arguments = {
        "prompt": job.prompt,
        "image_size": job.image_size,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "seed": job.seed,
        "sync_mode": SYNC_MODE,
        "num_images": NUM_IMAGES,
        "enable_safety_checker": ENABLE_SAFETY_CHECKER,
        "output_format": OUTPUT_FORMAT,
        "acceleration": ACCELERATION,
        "enable_prompt_expansion": ENABLE_PROMPT_EXPANSION,
    }

    async with semaphore:
        try:
            handler = await fal_client.submit_async(MODEL_ID, arguments=arguments)
            result = await handler.get()

            images = result_get(result, "images", [])
            if not images:
                raise RuntimeError("fal result does not contain images")

            image = images[0]
            image_url = result_get(image, "url")
            if not image_url:
                raise RuntimeError("fal image result does not contain url")

            await download_image(client, image_url, job.output_path)

            return MetadataRow(
                image_id=job.image_id,
                local_path=str(job.output_path),
                model_id=MODEL_ID,
                prompt_id=job.prompt_id,
                category=job.category,
                prompt=job.prompt,
                seed=job.seed,
                returned_seed=result_get(result, "seed"),
                image_size=job.image_size,
                width=result_get(image, "width"),
                height=result_get(image, "height"),
                content_type=result_get(image, "content_type"),
                source_url=image_url,
                request_id=getattr(handler, "request_id", None),
                has_nsfw_concepts=json.dumps(
                    result_get(result, "has_nsfw_concepts", [])
                ),
                timings=json.dumps(
                    result_get(result, "timings", {}), ensure_ascii=False
                ),
                num_inference_steps=NUM_INFERENCE_STEPS,
                output_format=OUTPUT_FORMAT,
                enable_prompt_expansion=ENABLE_PROMPT_EXPANSION,
                acceleration=ACCELERATION,
                num_images=NUM_IMAGES,
                sync_mode=SYNC_MODE,
                enable_safety_checker=ENABLE_SAFETY_CHECKER,
                status="ok",
                error="",
            )
        except Exception as exc:
            logger.exception("Generation failed for image_id=%s", job.image_id)
            return MetadataRow(
                image_id=job.image_id,
                local_path=str(job.output_path),
                model_id=MODEL_ID,
                prompt_id=job.prompt_id,
                category=job.category,
                prompt=job.prompt,
                seed=job.seed,
                returned_seed="",
                image_size=job.image_size,
                width="",
                height="",
                content_type="",
                source_url="",
                request_id="",
                has_nsfw_concepts="",
                timings="",
                num_inference_steps=NUM_INFERENCE_STEPS,
                output_format=OUTPUT_FORMAT,
                enable_prompt_expansion=ENABLE_PROMPT_EXPANSION,
                acceleration=ACCELERATION,
                num_images=NUM_IMAGES,
                sync_mode=SYNC_MODE,
                enable_safety_checker=ENABLE_SAFETY_CHECKER,
                status="error",
                error=str(exc),
            )


def write_metadata(rows: list[MetadataRow]) -> None:
    """Write generation metadata to CSV.

    Args:
        rows: Metadata rows.
    """
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(asdict(rows[0]).keys())
    with METADATA_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


async def run_generation(concurrency: int) -> None:
    """Run all generation jobs.

    Args:
        concurrency: Maximum concurrent fal requests.
    """
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs()
    semaphore = asyncio.Semaphore(concurrency)

    logger.info("Starting %s jobs with concurrency=%s", len(jobs), concurrency)

    async with httpx.AsyncClient(timeout=180.0) as client:
        tasks = [generate_one(job, client, semaphore) for job in jobs]
        rows = await asyncio.gather(*tasks)

    rows.sort(key=lambda row: (row.prompt_id, row.seed))
    write_metadata(rows)

    ok_count = sum(row.status == "ok" for row in rows)
    error_count = len(rows) - ok_count
    logger.info("Finished generation: ok=%s error=%s", ok_count, error_count)
    logger.info("Images directory: %s", IMAGE_DIR)
    logger.info("Metadata CSV: %s", METADATA_PATH)


@app.command()
def main(
    concurrency: Annotated[
        int,
        typer.Argument(
            min=1,
            help="Maximum number of concurrent fal requests.",
        ),
    ] = 10,
) -> None:
    """Generate 100 Fal Z-Image Turbo images."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    asyncio.run(run_generation(concurrency))


if __name__ == "__main__":
    app()
