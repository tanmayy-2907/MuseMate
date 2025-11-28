import ExpandingPanels from "@/components/ExpandingPanels";
import HeroSection from "@/components/HeroSection";
import HowItWorks from "@/components/HowItWorks";
import FinalCTA from "@/components/FinalCTA";
import Footer from "@/components/Footer";
import ThemeToggle from "@/components/ThemeToggle";
import introimg1 from "@/assets/introimg1.png";
import introimg2 from "@/assets/introimg2.png";
import introimg3 from "@/assets/introimg3.png";
import introimg4 from "@/assets/introimg4.png";
import introimg5 from "@/assets/introimg5.png";
import introimg6 from "@/assets/introimg6.png";
import introimg7 from "@/assets/introimg7.png";
import introimg8 from "@/assets/introimg8.png";

const artworks = [
  {
    image: introimg1,
    title: "Explore the world through art",
    description: ""
  },
  {
    image: introimg2,
    title: "Crafted with passion",
    description: ""
  },
  {
    image: introimg3,
    title: "Where Imagination Meets Reality",
    description: ""
  },
  {
    image: introimg4,
    title: "A Window Into History",
    description: ""
  },
  {
    image: introimg5,
    title: "Wander with Colors",
    description: ""
  },
  {
    image: introimg6,
    title: "Art that speaks",
    description: ""
  },
  {
    image: introimg7,
    title: "Dive into Nature",
    description: ""
  },
  {
    image: introimg8,
    title: "Know popular artist over the world",
    description: ""
  }
];

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <ThemeToggle />
      {/* Hero Section */}
      <HeroSection />

      {/* How It Works Section */}
      <HowItWorks />

      {/* Gallery Section */}
      <section className="py-12 px-4 bg-gradient-to-b from-background via-card/20 to-background">
        <div className="max-w-[98%] mx-auto">
          <div className="text-center mb-8">
            <h2 className="text-4xl md:text-5xl font-bold text-foreground mb-4">
              Gallery of <span className="text-primary">Inspiration</span>
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Watch as each masterpiece comes to life
            </p>
          </div>

          <ExpandingPanels artworks={artworks} />
        </div>
      </section>

      {/* Final CTA Section */}
      <FinalCTA />

      {/* Footer */}
      <Footer />
    </div>
  );
};

export default Index;
