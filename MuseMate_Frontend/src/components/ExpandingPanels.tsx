import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";

interface Artwork {
  image: string;
  title: string;
  description: string;
}

interface ExpandingPanelsProps {
  artworks: Artwork[];
}

const ExpandingPanels = ({ artworks }: ExpandingPanelsProps) => {
  const [activeIndex, setActiveIndex] = useState(0);

  // Auto-rotate through panels every 2 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % artworks.length);
    }, 2000);

    return () => clearInterval(interval);
  }, [artworks.length]);

  return (
    <div className="flex h-[80vh] min-h-[600px] gap-2 w-full">
      {artworks.map((artwork, index) => {
        const isActive = index === activeIndex;

        return (
          <div
            key={index}
            className={cn(
              "relative overflow-hidden rounded-lg cursor-pointer transition-all duration-700 ease-out",
              isActive ? "flex-[5]" : "flex-[0.5]"
            )}
            onClick={() => setActiveIndex(index)}
            style={{
              backgroundImage: `url(${artwork.image})`,
              backgroundSize: "cover",
              backgroundPosition: "center",
            }}
          >
            {/* Dark overlay for inactive panels */}
            <div
              className={cn(
                "absolute inset-0 bg-background/70 transition-opacity duration-700",
                isActive ? "opacity-0" : "opacity-100"
              )}
            />

            {/* Gradient overlay for active panel */}
            {isActive && (
              <div className="absolute inset-0 bg-gradient-to-t from-background via-background/40 to-transparent" />
            )}

            {/* Content - only visible when active */}
            {isActive && (
              <div className="absolute inset-0 flex items-end p-8 animate-fade-in">
                <div className="bg-background/80 backdrop-blur-md rounded-2xl p-8 w-full max-w-2xl border border-primary/30">
                  <h3 className="text-4xl md:text-5xl font-bold text-primary">
                    {artwork.title}
                  </h3>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default ExpandingPanels;
