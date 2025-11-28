import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Palette, Users } from "lucide-react";

interface RecommendationsProps {
  recommendations: {
    similarArtworks: Array<{
      name: string;
      artist: string;
      year: string;
      reason: string;
    }>;
    similarArtists: Array<{
      name: string;
      period: string;
      style: string;
      reason: string;
    }>;
  };
}

export const Recommendations = ({ recommendations }: RecommendationsProps) => {
  return (
    <div className="space-y-8">
      {/* Similar Artworks */}
      <Card className="p-8 space-y-6">
        <div className="flex items-center gap-3">
          <Palette className="w-6 h-6 text-primary" />
          <h3 className="text-2xl md:text-3xl font-bold">Similar Artworks</h3>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {recommendations.similarArtworks.map((artwork, index) => (
            <Card
              key={index}
              className="p-6 space-y-3 hover:border-primary/50 transition-all duration-300 animate-fade-in"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div>
                <h4 className="text-lg font-bold mb-1">{artwork.name}</h4>
                <p className="text-sm text-muted-foreground">
                  {artwork.artist} • {artwork.year}
                </p>
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {artwork.reason}
              </p>
            </Card>
          ))}
        </div>
      </Card>

      {/* Similar Artists */}
      <Card className="p-8 space-y-6">
        <div className="flex items-center gap-3">
          <Users className="w-6 h-6 text-primary" />
          <h3 className="text-2xl md:text-3xl font-bold">Similar Artists</h3>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {recommendations.similarArtists.map((artist, index) => (
            <Card
              key={index}
              className="p-6 space-y-3 hover:border-primary/50 transition-all duration-300 animate-fade-in"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div>
                <h4 className="text-lg font-bold mb-1">{artist.name}</h4>
                <div className="flex flex-wrap gap-2 mt-2">
                  <Badge variant="secondary" className="text-xs">{artist.period}</Badge>
                  <Badge variant="outline" className="text-xs">{artist.style}</Badge>
                </div>
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {artist.reason}
              </p>
            </Card>
          ))}
        </div>
      </Card>
    </div>
  );
};
