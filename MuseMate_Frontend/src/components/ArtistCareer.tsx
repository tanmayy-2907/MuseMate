import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { User } from "lucide-react";

interface ArtistCareerProps {
  artist: {
    name: string;
    timeline: Array<{
      year: string;
      event: string;
      description: string;
      imageUrl?: string;
    }>;
  };
}

export const ArtistCareer = ({ artist }: ArtistCareerProps) => {
  return (
    <Card className="p-8 space-y-6">
      <div className="flex items-center gap-3">
        <User className="w-6 h-6 text-primary" />
        <h3 className="text-2xl md:text-3xl font-bold">
          <span className="text-primary">{artist.name}</span>'s Career Timeline
        </h3>
      </div>

      <Separator />

      <div className="relative space-y-6">
        {/* Timeline line */}
        <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-border hidden md:block" />

        {artist.timeline.map((item, index) => (
          <div
            key={index}
            className="relative animate-fade-in"
            style={{ animationDelay: `${index * 0.1}s` }}
          >
            {/* Timeline dot */}
            <div className="absolute left-2.5 top-2 w-3 h-3 rounded-full bg-primary border-2 border-background hidden md:block" />

            <div className="md:ml-12 space-y-3">
              <div className="flex items-baseline gap-3 flex-wrap">
                <Badge variant="secondary" className="text-base">
                  {item.year}
                </Badge>
                <h4 className="text-xl font-semibold">{item.event}</h4>
              </div>
              
              {item.imageUrl && (
                <div className="rounded-lg overflow-hidden border border-border max-w-md">
                  <img 
                    src={item.imageUrl} 
                    alt={item.event}
                    className="w-full h-auto object-cover"
                    loading="lazy"
                  />
                </div>
              )}
              
              <p className="text-muted-foreground leading-relaxed">
                {item.description}
              </p>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
};
