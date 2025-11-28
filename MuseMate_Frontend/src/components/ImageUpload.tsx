import { useCallback, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Upload, Image as ImageIcon, Loader2, Sparkles } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import uploadPlaceholder from "@/assets/upload-placeholder.png";

interface ImageUploadProps {
  onImageUpload: (imageData: string) => void;
  uploadedImage: string | null;
  isProcessing: boolean;
  onRecognize: () => void;
}

export const ImageUpload = ({
  onImageUpload,
  uploadedImage,
  isProcessing,
  onRecognize
}: ImageUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const { toast } = useToast();

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = (e) => {
          onImageUpload(e.target?.result as string);
        };
        reader.readAsDataURL(file);
      } else {
        toast({
          title: "Invalid file",
          description: "Please upload an image file",
          variant: "destructive",
        });
      }
    },
    [onImageUpload, toast]
  );

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        onImageUpload(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="text-center space-y-4">
        <h2 className="text-4xl md:text-5xl font-bold">
          Upload Your <span className="text-gradient">Artwork</span>
        </h2>
        <p className="text-lg text-muted-foreground">
          Drop an image or select from your device to begin the discovery
        </p>
      </div>

      <Card
        className={`relative p-12 border-2 border-dashed transition-all duration-300 ${
          isDragging
            ? "border-primary bg-primary/5 scale-105"
            : "border-border hover:border-primary/50"
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {!uploadedImage ? (
          <div className="text-center space-y-6">
            <div className="mx-auto w-32 h-32 rounded-2xl bg-card/50 flex items-center justify-center border border-border">
              <img 
                src={uploadPlaceholder} 
                alt="Upload" 
                className="w-20 h-20 opacity-50"
              />
            </div>

            <div className="space-y-2">
              <p className="text-xl font-medium">Drop your artwork here</p>
              <p className="text-muted-foreground">or click to browse</p>
            </div>

            <input
              type="file"
              accept="image/*"
              onChange={handleFileInput}
              className="hidden"
              id="file-upload"
            />
            <label htmlFor="file-upload">
              <Button size="lg" className="gallery-glow" asChild>
                <span>
                  <Upload className="mr-2 h-5 w-5" />
                  Select Image
                </span>
              </Button>
            </label>

            <p className="text-sm text-muted-foreground">
              Supports JPG, PNG, WEBP (Max 10MB)
            </p>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="relative rounded-xl overflow-hidden border border-border">
              <img
                src={uploadedImage}
                alt="Uploaded artwork"
                className="w-full h-auto max-h-[500px] object-contain bg-card"
              />
            </div>

            <div className="flex gap-4 justify-center">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileInput}
                className="hidden"
                id="file-replace"
              />
              <label htmlFor="file-replace">
                <Button variant="outline" asChild>
                  <span>
                    <ImageIcon className="mr-2 h-4 w-4" />
                    Change Image
                  </span>
                </Button>
              </label>

              <Button
                size="lg"
                onClick={onRecognize}
                disabled={isProcessing}
                className="gallery-glow min-w-[200px]"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Sparkles className="mr-2 h-5 w-5" />
                    Recognize Artwork
                  </>
                )}
              </Button>
            </div>
          </div>
        )}
      </Card>

      {isProcessing && (
        <div className="text-center space-y-4 animate-pulse">
          <p className="text-lg text-muted-foreground">
            Our AI is analyzing the artwork...
          </p>
          <div className="flex justify-center gap-2">
            <div className="w-2 h-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: "0ms" }} />
            <div className="w-2 h-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: "150ms" }} />
            <div className="w-2 h-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: "300ms" }} />
          </div>
        </div>
      )}
    </div>
  );
};
