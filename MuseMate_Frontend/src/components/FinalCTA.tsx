import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";

const FinalCTA = () => {
  const navigate = useNavigate();
  
  return (
    <section className="py-32 px-4 relative overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-background via-card/50 to-background pointer-events-none" />
      
      <div className="relative z-10 max-w-4xl mx-auto text-center">
        <h2 className="text-4xl md:text-6xl font-bold mb-6 animate-fade-in">
          Ready to Unlock the <br />
          <span className="text-primary">Secrets of Art?</span>
        </h2>

        <p className="text-lg md:text-xl text-muted-foreground mb-10 animate-fade-in" style={{ animationDelay: "0.1s" }}>
          Join thousands discovering the stories behind masterpieces
        </p>

        <Button 
          size="lg" 
          className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold px-8 py-6 text-lg rounded-full shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 animate-fade-in"
          style={{ animationDelay: "0.2s" }}
          onClick={() => navigate("/recognize")}
        >
          Start Exploring Now
        </Button>
      </div>
    </section>
  );
};

export default FinalCTA;
