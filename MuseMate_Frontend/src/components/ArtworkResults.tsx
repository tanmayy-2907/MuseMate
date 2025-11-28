import { useState } from "react";
import { Button } from "@/components/ui/button";
import { CheckCircle2, RotateCcw, Clock, Palette, Users, ArrowLeft } from "lucide-react";
import type { ArtworkInfo } from "@/pages/Recognize";
import { ArtworkDetails } from "./ArtworkDetails";
import { ArtistCareer } from "./ArtistCareer";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface ArtworkResultsProps {
  artworkInfo: ArtworkInfo;
  uploadedImage: string;
  onExploreMore: () => void;
  onReset: () => void;
}

type ExploreView = 'main' | 'timeline' | 'artworks' | 'artists';

export const ArtworkResults = ({
  artworkInfo,
  uploadedImage,
  onReset
}: ArtworkResultsProps) => {
  const [currentView, setCurrentView] = useState<ExploreView>('main');

  const handleBackToMain = () => setCurrentView('main');

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Success Banner */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-primary animate-scale-in">
          <CheckCircle2 className="w-6 h-6" />
          <p className="text-lg font-medium">Artwork Successfully Identified!</p>
        </div>
        <Button
          variant="outline"
          onClick={onReset}
          className="gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          Try Another
        </Button>
      </div>

      {currentView === 'main' && (
        <>
          {/* Artwork Details */}
          <div className="animate-fade-in">
            <ArtworkDetails 
              artwork={artworkInfo.artwork}
              uploadedImage={uploadedImage}
              confidence={artworkInfo.confidence}
            />
          </div>

          {/* Explore More Section */}
          <div className="animate-fade-in" style={{ animationDelay: "0.1s" }}>
            <Card className="p-8 text-center space-y-6 bg-gradient-to-br from-primary/5 to-primary/10">
              <div className="space-y-2">
                <h3 className="text-2xl md:text-3xl font-bold">
                  Discover More About This Masterpiece
                </h3>
                <p className="text-muted-foreground text-lg">
                  Dive deeper into the artist's journey, explore similar artworks, or discover artists with comparable styles
                </p>
              </div>

              <div className="grid md:grid-cols-3 gap-4 max-w-4xl mx-auto">
                <Button
                  size="lg"
                  onClick={() => setCurrentView('timeline')}
                  className="gap-2 h-auto py-4 flex-col"
                >
                  <Clock className="w-6 h-6" />
                  <span className="font-semibold">Artist Timeline</span>
                  <span className="text-xs opacity-80">Explore their creative journey</span>
                </Button>

                <Button
                  size="lg"
                  onClick={() => setCurrentView('artworks')}
                  className="gap-2 h-auto py-4 flex-col"
                >
                  <Palette className="w-6 h-6" />
                  <span className="font-semibold">Similar Artworks</span>
                  <span className="text-xs opacity-80">Find related masterpieces</span>
                </Button>

                <Button
                  size="lg"
                  onClick={() => setCurrentView('artists')}
                  className="gap-2 h-auto py-4 flex-col"
                >
                  <Users className="w-6 h-6" />
                  <span className="font-semibold">Similar Artists</span>
                  <span className="text-xs opacity-80">Discover comparable creators</span>
                </Button>
              </div>
            </Card>
          </div>
        </>
      )}

      {currentView === 'timeline' && (
        <div className="animate-fade-in">
          <Button
            variant="outline"
            onClick={handleBackToMain}
            className="gap-2 mb-6"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Details
          </Button>
          <ArtistCareer artist={artworkInfo.artist} />
        </div>
      )}

      {currentView === 'artworks' && (
        <div className="animate-fade-in">
          <Button
            variant="outline"
            onClick={handleBackToMain}
            className="gap-2 mb-6"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Details
          </Button>
          <Card className="p-8 space-y-6">
            <div className="flex items-center gap-3">
              <Palette className="w-6 h-6 text-primary" />
              <h3 className="text-2xl md:text-3xl font-bold">Similar Artworks</h3>
            </div>

            {artworkInfo.recommendations.similarArtworks.length > 0 ? (
              <div className="grid md:grid-cols-2 gap-6">
                {artworkInfo.recommendations.similarArtworks.map((artwork, index) => (
                  <Card key={index} className="p-6 space-y-3">
                    {artwork.imageUrl && (
                      <div className="rounded-lg overflow-hidden border border-border mb-3">
                        <img 
                          src={artwork.imageUrl} 
                          alt={artwork.name}
                          className="w-full h-48 object-cover"
                          loading="lazy"
                        />
                      </div>
                    )}
                    <div className="space-y-1">
                      <h4 className="text-xl font-semibold">{artwork.name}</h4>
                      <div className="flex gap-2 text-sm text-muted-foreground">
                        <span>{artwork.artist}</span>
                        <span>•</span>
                        <span>{artwork.year}</span>
                      </div>
                    </div>
                    <p className="text-muted-foreground">{artwork.reason}</p>
                  </Card>
                ))}
              </div>
            ) : (
              <p className="text-center text-muted-foreground py-8">
                No similar artworks available at the moment.
              </p>
            )}
          </Card>
        </div>
      )}

      {currentView === 'artists' && (
        <div className="animate-fade-in">
          <Button
            variant="outline"
            onClick={handleBackToMain}
            className="gap-2 mb-6"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Details
          </Button>
          <Card className="p-8 space-y-6">
            <div className="flex items-center gap-3">
              <Users className="w-6 h-6 text-primary" />
              <h3 className="text-2xl md:text-3xl font-bold">Similar Artists</h3>
            </div>

            {artworkInfo.recommendations.similarArtists.length > 0 ? (
              <div className="grid md:grid-cols-2 gap-6">
                {artworkInfo.recommendations.similarArtists.map((artist, index) => (
                  <Card key={index} className="p-6 space-y-3">
                    {artist.imageUrl && (
                      <div className="rounded-lg overflow-hidden border border-border mb-3">
                        <img 
                          src={artist.imageUrl} 
                          alt={artist.name}
                          className="w-full h-48 object-cover"
                          loading="lazy"
                        />
                      </div>
                    )}
                    <div className="space-y-2">
                      <h4 className="text-xl font-semibold">{artist.name}</h4>
                      <div className="flex gap-2 flex-wrap">
                        <Badge variant="secondary">{artist.period}</Badge>
                        <Badge variant="outline">{artist.style}</Badge>
                      </div>
                    </div>
                    <p className="text-muted-foreground">{artist.reason}</p>
                  </Card>
                ))}
              </div>
            ) : (
              <p className="text-center text-muted-foreground py-8">
                No similar artists available at the moment.
              </p>
            )}
          </Card>
        </div>
      )}
    </div>
  );
};
