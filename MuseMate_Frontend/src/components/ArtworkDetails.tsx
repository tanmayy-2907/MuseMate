import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Calendar, Palette, FileText } from "lucide-react";

interface ArtworkDetailsProps {
  artwork: {
    name: string;
    timeRange: string;
    location: string;
    types: string[];
    description: string;
  };
  uploadedImage: string;
  confidence: number;
}

export const ArtworkDetails = ({ artwork, uploadedImage, confidence }: ArtworkDetailsProps) => {
  return (
    <Card className="p-8 space-y-6">
      <div className="grid md:grid-cols-2 gap-8">
        {/* Image */}
        <div className="space-y-4">
          <div className="relative rounded-xl overflow-hidden border border-border">
            <img
              src={uploadedImage}
              alt={artwork.name}
              className="w-full h-auto object-contain bg-card"
            />
          </div>
          <Badge variant="secondary" className="text-sm">
            Confidence: {(confidence * 100).toFixed(2)}%
          </Badge>
        </div>

        {/* Details */}
        <div className="space-y-6">
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-2">{artwork.name}</h2>
          </div>

          <div className="space-y-4">
            {artwork.timeRange !== "Unknown" && (
              <div className="flex items-start gap-3">
                <Calendar className="w-5 h-5 text-primary mt-1 flex-shrink-0" />
                <div>
                  <p className="font-medium text-sm text-muted-foreground">Time Period</p>
                  <p className="text-lg">{artwork.timeRange}</p>
                </div>
              </div>
            )}

            {artwork.types.length > 0 && !artwork.types.includes("Unknown") && (
              <div className="flex items-start gap-3">
                <Palette className="w-5 h-5 text-primary mt-1 flex-shrink-0" />
                <div>
                  <p className="font-medium text-sm text-muted-foreground">Classification</p>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {artwork.types.map((type, index) => (
                      <Badge key={index} variant="outline">{type}</Badge>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {artwork.description !== "Unknown" && (
              <div className="flex items-start gap-3">
                <FileText className="w-5 h-5 text-primary mt-1 flex-shrink-0" />
                <div>
                  <p className="font-medium text-sm text-muted-foreground mb-2">Description</p>
                  <p className="text-muted-foreground leading-relaxed">
                    {artwork.description}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
};
