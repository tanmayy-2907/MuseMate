const steps = [
  {
    number: "01",
    title: "Upload Your Image",
    description: "Simply drag and drop or select any artwork image from your device"
  },
  {
    number: "02",
    title: "AI Recognition",
    description: "Our advanced AI instantly analyzes and identifies the artwork and artist"
  },
  {
    number: "03",
    title: "Explore & Discover",
    description: "Dive into detailed information and explore the complete artist timeline"
  }
];

const HowItWorks = () => {
  return (
    <section className="py-20 px-4">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-4xl md:text-5xl font-bold text-center mb-16">
          How It <span className="text-primary">Works</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {steps.map((step, index) => (
            <div
              key={index}
              className="group relative bg-card border border-border rounded-2xl p-8 transition-all duration-500 hover:border-primary/50 hover:shadow-xl hover:shadow-primary/10 animate-fade-in"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              {/* Number */}
              <div className="text-6xl font-bold text-primary/20 mb-4 transition-colors duration-300 group-hover:text-primary/30">
                {step.number}
              </div>

              {/* Title */}
              <h3 className="text-2xl font-bold mb-4 text-foreground">
                {step.title}
              </h3>

              {/* Description */}
              <p className="text-muted-foreground leading-relaxed">
                {step.description}
              </p>

              {/* Hover glow effect */}
              <div className="absolute inset-0 rounded-2xl bg-primary/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
