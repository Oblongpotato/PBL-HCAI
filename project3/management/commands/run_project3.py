from django.core.management.base import BaseCommand

from project3 import experiments


class Command(BaseCommand):
    help = "Run the project 3 experiments and write the results the page displays."

    def handle(self, *args, **options):
        self.stdout.write("Running experiments, this takes a couple of minutes...")
        results = experiments.run_all()
        self.stdout.write(self.style.SUCCESS(f"Wrote {experiments.RESULTS_FILE}"))
        for section in results:
            self.stdout.write(f"  {section}")
