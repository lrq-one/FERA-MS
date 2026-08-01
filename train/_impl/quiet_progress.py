import re
import sys


ansi_pattern = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"
)

epoch_pattern = re.compile(
    r"Epoch\s+(\d+):"
)

stage_heading_pattern = re.compile(
    r"^(?:"
    r"STAGE\d+_[A-Z0-9_-]+"
    r"|refined variant[AC][A-Z0-9 _-]*"
    r"|R(?:146|147|149B|150A|150B|153|154|160|172D|184B)"
    r"[A-Z0-9 _-]*"
    r")$"
)

important_tokens = (
    "Metric ",
    "New best score",
    "EarlyStopping",
    "`Trainer.fit` stopped",
    "validation cosine",
    "validation jss",
    "best_val",
    "test cosine",
    "test jss",
    "test_used_for_selection",
    "mainline complete",
    "Traceback",
    "Error",
    "Exception",
    "CUDA out of memory",
    "KeyboardInterrupt",
    "FAILED",
)

training_started = False
expect_new_stage = False

global_epoch = 0
last_local_epoch = None

buffer = ""


def write_output(text, delimiter):
    sys.stdout.write(text)
    sys.stdout.write(delimiter)
    sys.stdout.flush()


def process(record, delimiter):
    global training_started
    global expect_new_stage
    global global_epoch
    global last_local_epoch

    clean = ansi_pattern.sub(
        "",
        record,
    ).strip()

    if not training_started:
        epoch_match = epoch_pattern.search(
            clean
        )

        if epoch_match is None:
            write_output(
                record,
                delimiter,
            )
            return

        training_started = True

    heading = (
        clean.upper()
        .strip()
        .strip("= []")
        .strip()
    )

    if (
        heading
        and stage_heading_pattern.fullmatch(
            heading
        )
    ):
        expect_new_stage = True
        return

    epoch_match = epoch_pattern.search(
        clean
    )

    if epoch_match is not None:
        local_epoch = int(
            epoch_match.group(1)
        )

        if last_local_epoch is None:
            global_epoch = local_epoch + 1
            last_local_epoch = local_epoch
            expect_new_stage = False

        elif expect_new_stage or local_epoch < last_local_epoch:
            global_epoch += 1
            last_local_epoch = local_epoch
            expect_new_stage = False

        elif local_epoch != last_local_epoch:
            global_epoch += local_epoch - last_local_epoch
            last_local_epoch = local_epoch

        rewritten = epoch_pattern.sub(
            f"Epoch {global_epoch}:",
            record,
            count=1,
        )

        write_output(
            rewritten,
            delimiter,
        )
        return

    if any(
        token.lower() in clean.lower()
        for token in important_tokens
    ):
        write_output(
            record,
            delimiter,
        )


try:
    while True:
        chunk = sys.stdin.buffer.read1(
            65536
        )

        if not chunk:
            break

        buffer += chunk.decode(
            "utf-8",
            errors="replace",
        )

        while True:
            carriage = buffer.find("\r")
            newline = buffer.find("\n")

            positions = [
                position
                for position in (
                    carriage,
                    newline,
                )
                if position >= 0
            ]

            if not positions:
                break

            position = min(positions)
            delimiter = buffer[position]

            record = buffer[:position]
            buffer = buffer[position + 1:]

            process(
                record,
                delimiter,
            )

    if buffer:
        process(
            buffer,
            "",
        )

except KeyboardInterrupt:
    pass
