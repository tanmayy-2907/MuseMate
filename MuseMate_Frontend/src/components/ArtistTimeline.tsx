import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowLeft, Calendar } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";

interface ArtistTimelineProps {
  artist: {
    name: string;
    timeline: Array<{
      year: string;
      event: string;
      description: string;
    }>;
  };
  onBack: () => void;
}

export const ArtistTimeline = ({ artist, onBack }: ArtistTimelineProps) => {
  return (
    <div className="max-w-6xl mx-auto space-y-8 animate-fade-in">
      {/* Header */}
      <div className="space-y-4">
        <Button
          variant="ghost"
          onClick={onBack}
          className="gap-2"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Results
        </Button>

        <div className="space-y-2">
          <h2 className="text-4xl md:text-5xl font-bold">
            <span className="text-gradient">{artist.name}</span>'s Journey
          </h2>
          <p className="text-lg text-muted-foreground">
            Explore the evolution of a master through their most significant works
          </p>
        </div>
      </div>

      {/* Timeline */}
      <div className="relative">
        {/* Vertical Line */}
        <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gradient-to-b from-primary via-primary/50 to-transparent hidden md:block" />

        <ScrollArea className="h-[800px]">
          <div className="space-y-8 pb-8">
            {artist.timeline.map((item, index) => (
              <div
                key={index}
                className="relative animate-slide-up"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                {/* Timeline Dot */}
                <div className="absolute left-6 top-6 w-5 h-5 rounded-full bg-primary border-4 border-background hidden md:block z-10" />

                <Card className="md:ml-20 p-6 hover:border-primary/50 transition-all duration-300">
                  <div className="space-y-4">
                    <div className="flex items-start justify-between gap-4">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <Calendar className="w-4 h-4 text-primary" />
                          <span className="text-2xl font-bold text-primary">
                            {item.year}
                          </span>
                        </div>
                        <h3 className="text-2xl font-bold">{item.event}</h3>
                      </div>
                    </div>

                    <p className="text-muted-foreground leading-relaxed">
                      {item.description}
                    </p>
                  </div>
                </Card>
              </div>
            ))}
          </div>
        </ScrollArea>
      </div>

      {/* Footer CTA */}
      <Card className="p-8 text-center bg-gradient-to-br from-card to-card/50 border-primary/20">
        <h3 className="text-2xl font-bold mb-2">Want to explore more artists?</h3>
        <p className="text-muted-foreground mb-4">
          Upload another artwork to begin a new discovery
        </p>
        <Button onClick={onBack} variant="outline">
          Start New Recognition
        </Button>
      </Card>
    </div>
  );
};
