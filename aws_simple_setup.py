#!/usr/bin/env python3

def simple_aws_setup():
    """Simple AWS setup without dependencies"""
    
    print("🚀 AWS HPO SETUP (No Dependencies)")
    print("=" * 35)
    
    print("💡 SIMPLEST OPTION: AWS EC2")
    print("• Launch big instance (96 CPUs)")
    print("• Upload your existing code")
    print("• Run HPO 10-20x faster")
    print("• Cost: $10-30 total")
    
    print(f"\n📋 STEPS:")
    print(f"1. Go to AWS Console → EC2")
    print(f"2. Launch Instance:")
    print(f"   - Type: c5.24xlarge (96 CPUs)")
    print(f"   - OS: Amazon Linux 2")
    print(f"   - Storage: 100GB")
    print(f"3. SSH to instance")
    print(f"4. Upload your code")
    print(f"5. Run: python run_hpo_resumable.py --resume")
    
    print(f"\n💰 COST ESTIMATE:")
    print(f"• c5.24xlarge: $4.60/hour")
    print(f"• Complete in: 4-8 hours")
    print(f"• Total cost: $18-37")
    print(f"• vs Local: 2-3 days remaining")
    
    print(f"\n🎯 ALTERNATIVE: Keep Local")
    print(f"• Your current 5 processes will finish")
    print(f"• Just wait 1-2 more days")
    print(f"• Cost: $0")
    
    choice = input(f"\nPrefer: (1) AWS EC2 setup, (2) Keep local, (3) More info? ")
    
    if choice == '1':
        print(f"\n🚀 AWS EC2 SETUP:")
        print(f"1. aws.amazon.com → Sign in")
        print(f"2. EC2 → Launch Instance")
        print(f"3. c5.24xlarge, Amazon Linux 2")
        print(f"4. Create key pair, download .pem")
        print(f"5. Launch → Note IP address")
        print(f"6. SSH: ssh -i key.pem ec2-user@IP")
        
    elif choice == '2':
        print(f"\n⏳ KEEPING LOCAL:")
        print(f"• Your 5 HPO processes will complete")
        print(f"• Check progress: python hpo_running_status.py")
        print(f"• Estimated: 1-2 days remaining")
        
    else:
        print(f"\n📖 MORE INFO:")
        print(f"• AWS EC2: Rent powerful computer by hour")
        print(f"• 96 CPUs vs your 24 CPUs = 4x faster")
        print(f"• Plus better optimization = 10-20x total")
        print(f"• Upload code, run same commands")

if __name__ == "__main__":
    simple_aws_setup()