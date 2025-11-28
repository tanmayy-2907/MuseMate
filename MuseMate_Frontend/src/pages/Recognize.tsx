import { useState } from "react";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { ImageUpload } from "@/components/ImageUpload";
import { ArtworkResults } from "@/components/ArtworkResults";
import { ArtistTimeline } from "@/components/ArtistTimeline";
import { useToast } from "@/hooks/use-toast";

export interface ArtworkInfo {
  artwork: {
    name: string;
    timeRange: string;
    location: string;
    types: string[];
    description: string;
  };
  artist: {
    name: string;
    timeline: Array<{
      year: string;
      event: string;
      description: string;
      imageUrl?: string;
    }>;
  };
  recommendations: {
    similarArtworks: Array<{
      name: string;
      artist: string;
      year: string;
      reason: string;
      imageUrl?: string;
    }>;
    similarArtists: Array<{
      name: string;
      period: string;
      style: string;
      reason: string;
      imageUrl?: string;
    }>;
  };
  confidence: number;
}

const Recognize = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [artworkInfo, setArtworkInfo] = useState<ArtworkInfo | null>(null);
  const [showTimeline, setShowTimeline] = useState(false);

  const handleImageUpload = (imageData: string) => {
    setUploadedImage(imageData);
    setArtworkInfo(null);
    setShowTimeline(false);
  };

  const handleRecognize = async () => {
    if (!uploadedImage) return;
    
    setIsProcessing(true);
    
    try {
      // Convert base64 to blob
      const response = await fetch(uploadedImage);
      const blob = await response.blob();
      
      // Create FormData to send to Flask API
      const formData = new FormData();
      formData.append('file', blob, 'artwork.jpg');
      
      // Call Flask API
      const apiResponse = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        body: formData,
      });
      
      if (!apiResponse.ok) {
        throw new Error('Failed to recognize artwork');
      }
      
      const data = await apiResponse.json();
      
      // Map Flask API response to ArtworkInfo format
      const mappedArtworkInfo: ArtworkInfo = {
        artwork: {
          name: data.predicted_title,
          timeRange: data.date,
          location: "Unknown",
          types: [data.genre, data.style].filter(Boolean),
          description: `This ${data.genre} was created by ${data.artist} in ${data.date}, representing the ${data.style} style.`,
        },
        artist: {
          name: data.artist,
          timeline: data.artist_timeline.map((item: any) => ({
            year: item.date,
            event: item.title,
            description: `A significant work from ${item.date}, showcasing the artist's evolving mastery and creative vision.`,
            imageUrl: item.image_url,
          })),
        },
        recommendations: {
          similarArtworks: data.similar_artworks.map((item: any) => ({
            name: item.title,
            artist: item.artist,
            year: "Unknown",
            reason: `This artwork shares stylistic similarities with the ${data.style} movement.`,
            imageUrl: item.image_url,
          })),
          similarArtists: data.similar_artists.map((item: any) => ({
            name: item.artist_name || item,
            period: data.date,
            style: data.style,
            reason: `${item.artist_name || item} worked in the ${data.style} style during a similar period.`,
            imageUrl: item.image_url,
          })),
        },
        confidence: data.confidence,
      };
      
      setArtworkInfo(mappedArtworkInfo);
      
      toast({
        title: "Artwork Recognized!",
        description: `Identified as "${data.predicted_title}" by ${data.artist}`,
      });
      
    } catch (error) {
      console.error('Error recognizing artwork:', error);
      toast({
        title: "Recognition Failed",
        description: "Unable to recognize the artwork. Please make sure the Flask API is running on port 5000.",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleExploreMore = () => {
    setShowTimeline(true);
  };

  const handleReset = () => {
    setUploadedImage(null);
    setArtworkInfo(null);
    setShowTimeline(false);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur-lg">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <Button
            variant="ghost"
            onClick={() => navigate("/")}
            className="gap-2"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Home
          </Button>
          <h1 
            className="text-2xl font-bold cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => navigate("/")}
          >
            <span className="text-primary">MuseMate</span>
          </h1>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-12">
        {!artworkInfo && !showTimeline && (
          <div className="animate-fade-in">
            <ImageUpload
              onImageUpload={handleImageUpload}
              uploadedImage={uploadedImage}
              isProcessing={isProcessing}
              onRecognize={handleRecognize}
            />
          </div>
        )}

        {artworkInfo && !showTimeline && (
          <div className="animate-fade-in">
            <ArtworkResults
              artworkInfo={artworkInfo}
              uploadedImage={uploadedImage!}
              onExploreMore={handleExploreMore}
              onReset={handleReset}
            />
          </div>
        )}

        {showTimeline && artworkInfo && (
          <div className="animate-fade-in">
            <ArtistTimeline
              artist={artworkInfo.artist}
              onBack={() => setShowTimeline(false)}
            />
          </div>
        )}
      </main>
    </div>
  );
};

export default Recognize;
