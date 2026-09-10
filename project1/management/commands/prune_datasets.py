"""Delete uploaded datasets that nobody is working on any more.

Uploads are kept so that visualisation and training can reuse them, which means the app
never removes them on its own. Run this to drop the old ones and the files behind them.
"""

from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from project1.models import Dataset


class Command(BaseCommand):
    help = "Delete uploaded datasets older than the given number of days."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=7)
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="List what would be deleted without deleting anything.",
        )

    def handle(self, *args, **options):
        cutoff = timezone.now() - timedelta(days=options["days"])
        stale = list(Dataset.objects.filter(uploaded_at__lt=cutoff))

        for dataset in stale:
            self.stdout.write(f"{dataset.uploaded_at:%Y-%m-%d} {dataset.name}")
            if not options["dry_run"]:
                # delete() on the field removes the file; delete() on the row removes the
                # training runs with it, through the cascade.
                dataset.file.delete(save=False)
                dataset.delete()

        verb = "would delete" if options["dry_run"] else "deleted"
        self.stdout.write(self.style.SUCCESS(f"{verb} {len(stale)} dataset(s)"))
