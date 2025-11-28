import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Eye, Sparkles, Image } from "lucide-react";
import { useNavigate } from "react-router-dom";
import heroGallery from "@/assets/hero-gallery.jpg";

const HeroSection = () => {
  const navigate = useNavigate();

  return (
    <section className="relative min-h-[90vh] flex items-center justify-center px-4 py-20 overflow-hidden">
      {/* Background image */}
      <div 
        className="absolute inset-0 bg-cover bg-center bg-no-repeat"
        style={{ backgroundImage: `url(${heroGallery})` }}
      />
      {/* Background gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-background/80 via-background/90 to-background pointer-events-none" />

      <div className="relative z-10 max-w-5xl mx-auto text-center">
        {/* Brand Name */}
        <div className="mb-4 animate-fade-in">
          <h1 className="text-6xl md:text-7xl lg:text-8xl font-bold text-primary mb-2">MuseMate</h1>
        </div>

        {/* Badge */}

        {/* Main heading */}
        <h2
          className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6 animate-fade-in"
          style={{ animationDelay: "0.2s" }}
        >
          Discover Art Through <span className="text-primary">Intelligence</span>
        </h2>

        {/* Subheading */}
        <p
          className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto mb-10 leading-relaxed animate-fade-in"
          style={{ animationDelay: "0.3s" }}
        >
          Upload any artwork and let AI unveil its secrets. Identify masterpieces, explore their stories, and journey
          through an artist's complete collection.
        </p>

        {/* CTA Button */}
        <Button
          size="lg"
          className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold px-8 py-6 text-lg rounded-full shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 animate-fade-in"
          style={{ animationDelay: "0.4s" }}
          onClick={() => navigate("/recognize")}
        >
          <Eye className="w-5 h-5 mr-2" />
          Begin Your Discovery
        </Button>

        {/* Feature badges */}
        <div
          className="flex flex-wrap items-center justify-center gap-4 mt-16 animate-fade-in"
          style={{ animationDelay: "0.5s" }}
        >
          <Badge variant="outline" className="px-4 py-2 text-sm bg-card/50 backdrop-blur-sm border-border">
            <Image className="w-4 h-4 mr-2 text-primary" />
            Instant Recognition
          </Badge>
          <Badge variant="outline" className="px-4 py-2 text-sm bg-card/50 backdrop-blur-sm border-border">
            <Sparkles className="w-4 h-4 mr-2 text-primary" />
            Rich Art History
          </Badge>
          <Badge variant="outline" className="px-4 py-2 text-sm bg-card/50 backdrop-blur-sm border-border">
            <Eye className="w-4 h-4 mr-2 text-primary" />
            Artist Collections
          </Badge>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
